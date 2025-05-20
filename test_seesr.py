'''
 * SeeSR: Towards Semantics-Aware Real-World Image Super-Resolution 
 * Modified from diffusers by Rongyuan Wu
 * 24/12/2023
'''
import os
import sys

from models.prompt_gan import PromptEnhancementGenerator
from transformers import CLIPTokenizer, CLIPTextModel
sys.path.append(os.getcwd())
import cv2
import glob
import argparse
import numpy as np
from PIL import Image
import torch
import torch.utils.checkpoint
from transformers import AutoTokenizer
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, DDPMScheduler
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available
from transformers import CLIPTextModel, CLIPTokenizer, CLIPImageProcessor
from models import prompt_gan
from pipelines.pipeline_seesr import StableDiffusionControlNetPipeline
from utils.misc import load_dreambooth_lora
from utils.wavelet_color_fix import wavelet_color_fix, adain_color_fix

from ram.models.ram_lora import ram
from ram import inference_ram as inference
from ram import get_transform

from typing import Mapping, Any
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from utils_text.phrase_generator import PhraseGenerator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

project_tag_embeds = nn.Linear(768, 1024).to(device)
decode_tag_embeds  = nn.Linear(1024, 768).to(device)

logger = get_logger(__name__, log_level="INFO")
bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

tensor_transforms = transforms.Compose([
                transforms.ToTensor(),
            ])

ram_transforms = transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

# 设定设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
project_tag_embeds = None

def load_state_dict_diffbirSwinIR(model: nn.Module, state_dict: Mapping[str, Any], strict: bool=False) -> None:
    state_dict = state_dict.get("state_dict", state_dict)
    
    is_model_key_starts_with_module = list(model.state_dict().keys())[0].startswith("module.")
    is_state_dict_key_starts_with_module = list(state_dict.keys())[0].startswith("module.")
    
    if (
        is_model_key_starts_with_module and
        (not is_state_dict_key_starts_with_module)
    ):
        state_dict = {f"module.{key}": value for key, value in state_dict.items()}
    if (
        (not is_model_key_starts_with_module) and
        is_state_dict_key_starts_with_module
    ):
        state_dict = {key[len("module."):]: value for key, value in state_dict.items()}
    
    model.load_state_dict(state_dict, strict=strict)


def load_seesr_pipeline(args, accelerator, enable_xformers_memory_efficient_attention):
    
    from models.controlnet import ControlNetModel
    from models.unet_2d_condition import UNet2DConditionModel

    # Load scheduler, tokenizer and models.
    
    scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_path, subfolder="scheduler")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_path, subfolder="text_encoder")
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_path, subfolder="tokenizer")
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_path, subfolder="vae")
    feature_extractor = CLIPImageProcessor.from_pretrained(f"{args.pretrained_model_path}/feature_extractor")
    unet = UNet2DConditionModel.from_pretrained(args.seesr_model_path, subfolder="unet")
    controlnet = ControlNetModel.from_pretrained(args.seesr_model_path, subfolder="controlnet")
    
    # Freeze vae and text_encoder
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)
    controlnet.requires_grad_(False)

    if enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            unet.enable_xformers_memory_efficient_attention()
            controlnet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    # Get the validation pipeline
    validation_pipeline = StableDiffusionControlNetPipeline(
        vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, feature_extractor=feature_extractor, 
        unet=unet, controlnet=controlnet, scheduler=scheduler, safety_checker=None, requires_safety_checker=False,
    )
    
    validation_pipeline._init_tiled_vae(encoder_tile_size=args.vae_encoder_tiled_size, decoder_tile_size=args.vae_decoder_tiled_size)

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move text_encode and vae to gpu and cast to weight_dtype
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device, dtype=weight_dtype)
    controlnet.to(accelerator.device, dtype=weight_dtype)

    return validation_pipeline

def load_tag_model(args, device='cuda'):
    
    model = ram(pretrained='preset/models/ram_swin_large_14m.pth',
                pretrained_condition=args.ram_ft_path,
                image_size=384,
                vit='swin_l')
    model.eval()
    model.to(device)
    
    return model

# phrase_generator = PhraseGenerator("F:/PyCharmProjects/SeeSR/lexicon.json")

def load_gan_model(args, device='cuda'):
    print("Loading GAN model...")
    model = PromptEnhancementGenerator(tag_dim=1024, image_embed_dim=512, hidden_dim=1024)
    if args.gan_model_path:
        print(f"Loading state dict from {args.gan_model_path}")
        model.load_state_dict(torch.load(args.gan_model_path, map_location=device))
    model.eval()
    model.to(device)
    print(f"Loaded GAN model: {model}")
    return model

def get_validation_prompt(
    args,
    image: Image.Image,
    tag_model,
    tokenizer,
    text_encoder,
    project_tag_embeds=None,
    decode_tag_embeds=None,
    gan_model=None,
    device='cuda',
    top_k: int = 5
):
    print(f"[DEBUG] tokenizer type: {type(tokenizer)}")
    print(f"[DEBUG] gan_model type: {type(gan_model)}")

    # 1) 预处理图像
    lq = tensor_transforms(image).unsqueeze(0).to(device)
    lq = ram_transforms(lq)

    # 2) 提取 RAM 特征
    raw_feats = tag_model.generate_image_embeds(lq)

    # 3) 池化到 [1, dim]
    if raw_feats.dim() == 4:
        ram_states = F.adaptive_avg_pool2d(raw_feats, 1).view(1, -1)
    elif raw_feats.dim() == 3:
        ram_states = raw_feats.mean(dim=1)
    else:
        ram_states = raw_feats

    # 4) 生成粗标签嵌入
    res = inference(lq, tag_model)
    inputs = tokenizer(
        res[0],
        padding="max_length",
        truncation=True,
        max_length=tokenizer.model_max_length,
        return_tensors="pt"
    ).to(device)
    text_feats = text_encoder(**inputs).last_hidden_state
    tag_embeds = text_feats.mean(dim=1)

    enhanced_embeds = None
    extra_words = None

    # 5) 如果有 GAN，就升维、跑 GAN，再解码 top_k 词
    if gan_model is not None and project_tag_embeds is not None:
        gan_input = project_tag_embeds(tag_embeds)  # [1,1024]
        enhanced_embeds = gan_model(gan_input, ram_states)  # [1,1024]

        # ==== DEBUG BEGIN ====
        # 1) 拿到 CLIP 的词向量矩阵
        emb_matrix = text_encoder.get_input_embeddings().weight  # [vocab_size, 768]
        # 先把 enhanced_embeds decode 回 768 维
        decoded = decode_tag_embeds(enhanced_embeds) if decode_tag_embeds else None
        # 2) 计算 GAN 输入 vs 输出 的向量差距
        delta_norm = (gan_input - enhanced_embeds).norm().item()
        # 3) 如果你能 decode 回 768，再对比 logits 差异
        if decode_tag_embeds is not None:
            # 解码回 768 维
            decoded = decode_tag_embeds(enhanced_embeds)  # [1,768]
            # 原始 tag_embeds （未映射时的词向量平均）
            logits_orig = tag_embeds @ emb_matrix.t()  # [1, vocab_size]
            # 新的 GAN 强化后的 logits
            logits_new = decoded @ emb_matrix.t()  # [1, vocab_size]
            avg_diff = (logits_new - logits_orig).abs().mean().item()
            print(f"[DEBUG] Δ-norm: {delta_norm:.4f}, avg-logit-diff: {avg_diff:.4f}")
        else:
            print(f"[DEBUG] Δ-norm: {delta_norm:.4f} (no decode branch)")
        # ==== DEBUG END ====

        if decode_tag_embeds is not None:
            decoded = decode_tag_embeds(enhanced_embeds)     # [1,768]
            emb_matrix = text_encoder.get_input_embeddings().weight  # [vocab,768]
            logits = decoded @ emb_matrix.t()               # [1, vocab]
            topk_ids = torch.topk(logits, top_k, dim=-1).indices[0].tolist()
            extra_words = tokenizer.decode(
                topk_ids, skip_special_tokens=True
            ).replace("  ", " ").strip()

            # **在这里打印 extra_words**
            print(f"[DEBUG] extra_words: {extra_words}")

        # 构造包含 extra_words 的 prompt
        if extra_words:
            prompt = f"{res[0]}, {extra_words}, {args.prompt},"
        else:
            prompt = f"{res[0]}, {args.prompt},"

        return prompt, ram_states, enhanced_embeds

    # —— 非 GAN 时也要返回三元组 ——
    prompt = f"{res[0]}, {args.prompt},"
    return prompt, ram_states, None


def main(args, enable_xformers_memory_efficient_attention=True,):
    global project_tag_embeds, decode_tag_embeds
    # 先设置设备与 accelerator
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    accelerator = Accelerator(mixed_precision=args.mixed_precision)

    project_tag_embeds = nn.Linear(768, 1024).to(accelerator.device)
    decode_tag_embeds = nn.Linear(1024, 768).to(accelerator.device)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    from transformers import AutoTokenizer, AutoModel
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    text_encoder = AutoModel.from_pretrained("bert-base-uncased").to(device)

    txt_path = os.path.join(args.output_dir, 'txt')
    os.makedirs(txt_path, exist_ok=True)

    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
    )

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # If use the enhacer, then initial it.
    if args.use_phrase_enhancement:
        global phrase_generator
        phrase_generator = PhraseGenerator(args.vocabulary_path)

    # Handle the output folder creation
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("SeeSR")

    pipeline = load_seesr_pipeline(args, accelerator, enable_xformers_memory_efficient_attention)
    tag_model = load_tag_model(args, accelerator.device)


    # 加载GAN模型
    if args.use_gan:
        gan_model = load_gan_model(args, accelerator.device)
        print(f"After initial loading, gan_model = {gan_model}")
        # project_tag_embeds = nn.Linear(768, 1024).to(accelerator.device)
        project_tag_embeds = project_tag_embeds
        decode_tag_embeds = decode_tag_embeds
        pipeline.gan_model = gan_model
    else:
        gan_model = None
        print("gan_model is set to None here!")
        project_tag_embeds = None
        decode_tag_embeds = None

    if accelerator.is_main_process:
        generator = torch.Generator(device=accelerator.device)
        if args.seed is not None:
            generator.manual_seed(args.seed)

        if os.path.isdir(args.image_path):
            image_names = sorted(glob.glob(f'{args.image_path}/*.*'))
        else:
            image_names = [args.image_path]

    for image_idx, image_name in enumerate(image_names[:]):
        print(f'================== process {image_idx} imgs... ===================')
        print(f"gan_model before processing: {gan_model}")
        validation_image = Image.open(image_name).convert("RGB")
        # 无论 args.use_gan，get_validation_prompt 都返回 (prompt, ram_states, enhanced_embeds)
        validation_prompt, ram_states, enhanced_embeds = get_validation_prompt(
            args, validation_image, tag_model,
            tokenizer, text_encoder,
            project_tag_embeds,decode_tag_embeds, gan_model,
            device=accelerator.device
        )
        validation_prompt = validation_prompt + args.added_prompt
        negative_prompt = args.negative_prompt

        if args.save_prompts:
            txt_save_path = f"{txt_path}/{os.path.basename(image_name).split('.')[0]}.txt"
            file = open(txt_save_path, "w")
            file.write(validation_prompt)
            file.close()
        print(f'{validation_prompt}')

        ori_width, ori_height = validation_image.size
        resize_flag = False
        rscale = args.upscale
        if ori_width < args.process_size//rscale or ori_height < args.process_size//rscale:
            scale = (args.process_size//rscale)/min(ori_width, ori_height)
            tmp_image = validation_image.resize((int(scale*ori_width), int(scale*ori_height)))

            validation_image = tmp_image
            resize_flag = True

        validation_image = validation_image.resize((validation_image.size[0]*rscale, validation_image.size[1]*rscale))
        validation_image = validation_image.resize((validation_image.size[0]//8*8, validation_image.size[1]//8*8))
        width, height = validation_image.size
        resize_flag = True #

        print(f'input size: {height}x{width}')

        for sample_idx in range(args.sample_times):
            os.makedirs(f'{args.output_dir}/sample{str(sample_idx).zfill(2)}/', exist_ok=True)

        if gan_model is not None:
            validation_prompt, ram_encoder_hidden_states, enhanced_tag_embeds = get_validation_prompt(
                args,
                validation_image,
                tag_model,tokenizer,
                text_encoder,
                project_tag_embeds,
                decode_tag_embeds,
                gan_model,
                device=accelerator.device)
        else:
            validation_prompt, ram_states, enhanced_embeds = get_validation_prompt(
                args, validation_image, tag_model, tokenizer, text_encoder, None, None,
            device = accelerator.device)

        tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_path, subfolder="tokenizer")
        text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_path, subfolder="text_encoder").to(
            accelerator.device)

        for sample_idx in range(args.sample_times):
            with torch.autocast("cuda"):
                # —— 1) 先用 CLIPTokenizer + CLIPTextModel 得到 prompt_embeds 和 negative_prompt_embeds ——
                input_ids = tokenizer(
                    validation_prompt,
                    padding="max_length",
                    truncation=True,
                    max_length=tokenizer.model_max_length,
                    return_tensors="pt",
                )
                prompt_embeds = text_encoder(input_ids.input_ids.to(accelerator.device)).last_hidden_state

                neg_ids = tokenizer(
                    negative_prompt,
                    padding="max_length",
                    truncation=True,
                    max_length=tokenizer.model_max_length,
                    return_tensors="pt",
                )
                negative_prompt_embeds = text_encoder(neg_ids.input_ids.to(accelerator.device)).last_hidden_state

                # —— 2) 把 ram_states 扩展到 [B, seq_len, dim] ——
                batch_size, seq_len, _ = prompt_embeds.shape
                dim = ram_states.shape[1]
                controlnet_embeds = ram_states.unsqueeze(1).expand(batch_size, seq_len, dim)

                # —— 3) 根据 enhanced_embeds 决定传不传 extra branch ——
                if enhanced_embeds is not None:
                    out = pipeline(
                        prompt=None,
                        prompt_embeds=prompt_embeds,
                        negative_prompt_embeds=negative_prompt_embeds,
                        image=validation_image,
                        ram_encoder_hidden_states=controlnet_embeds,
                        enhanced_tag_embeds=enhanced_embeds,
                        num_inference_steps=args.num_inference_steps,
                        guidance_scale=args.guidance_scale,
                        conditioning_scale=args.conditioning_scale,
                        start_point=args.start_point,
                        generator=generator,
                        height=height, width=width,
                        latent_tiled_size=args.latent_tiled_size,
                        latent_tiled_overlap=args.latent_tiled_overlap,
                    )
                else:
                    out = pipeline(
                        prompt=None,
                        prompt_embeds=prompt_embeds,
                        negative_prompt_embeds=negative_prompt_embeds,
                        image=validation_image,
                        ram_encoder_hidden_states=controlnet_embeds,
                        num_inference_steps=args.num_inference_steps,
                        guidance_scale=args.guidance_scale,
                        conditioning_scale=args.conditioning_scale,
                        start_point=args.start_point,
                        generator=generator,
                        height=height, width=width,
                        latent_tiled_size=args.latent_tiled_size,
                        latent_tiled_overlap=args.latent_tiled_overlap,
                    )

                image = out.images[0]

                if args.align_method == 'nofix':
                    image = image
                else:
                    if args.align_method == 'wavelet':
                        image = wavelet_color_fix(image, validation_image)
                    elif args.align_method == 'adain':
                        image = adain_color_fix(image, validation_image)

                if resize_flag: 
                    image = image.resize((ori_width*rscale, ori_height*rscale))
                    
                name, ext = os.path.splitext(os.path.basename(image_name))
                
                image.save(f'{args.output_dir}/sample{str(sample_idx).zfill(2)}/{name}.png')
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seesr_model_path", type=str, default="F:/PyCharmProjects/SeeSR/preset/models/seesr")
    parser.add_argument("--ram_ft_path", type=str, default="F:/PyCharmProjects/SeeSR/preset/models/DAPE.pth")
    parser.add_argument("--pretrained_model_path", type=str, default="F:/PyCharmProjects/SeeSR/preset/models/stable-diffusion-2-base/")
    parser.add_argument("--prompt", type=str, default="") # user can add self-prompt to improve the results
    parser.add_argument("--added_prompt", type=str, default="clean, high-resolution, 8k")
    parser.add_argument("--negative_prompt", type=str, default="dotted, noise, blur, lowres, smooth")
    parser.add_argument("--image_path", type=str, default="F:/PyCharmProjects/SeeSR/preset/datasets/test_datasets/")
    parser.add_argument("--output_dir", type=str, default="F:/PyCharmProjects/SeeSR/preset/datasets/output/")
    parser.add_argument("--mixed_precision", type=str, default="fp16") # no/fp16/bf16
    parser.add_argument("--guidance_scale", type=float, default=5.5)
    parser.add_argument("--conditioning_scale", type=float, default=1.0)
    parser.add_argument("--blending_alpha", type=float, default=1.0)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--process_size", type=int, default=512)
    parser.add_argument("--vae_decoder_tiled_size", type=int, default=224) # latent size, for 24G
    parser.add_argument("--vae_encoder_tiled_size", type=int, default=1024) # image size, for 13Gq
    parser.add_argument("--latent_tiled_size", type=int, default=96) 
    parser.add_argument("--latent_tiled_overlap", type=int, default=32) 
    parser.add_argument("--upscale", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample_times", type=int, default=1)
    parser.add_argument("--align_method", type=str, choices=['wavelet', 'adain', 'nofix'], default='adain')
    parser.add_argument("--start_steps", type=int, default=999) # defaults set to 999.
    parser.add_argument("--start_point", type=str, choices=['lr', 'noise'], default='lr') # LR Embedding Strategy, choose 'lr latent + 999 steps noise' as diffusion start point. 
    parser.add_argument("--save_prompts", action='store_true')
    parser.add_argument("--use_phrase_enhancement", action="store_true",
                        help="Enhance tags with attributive + noun phrases")
    parser.add_argument("--use_gan", action='store_true', help="Whether to use GAN for prompt enhancement")
    parser.add_argument("--gan_model_path", type=str, default='F:/PyCharmProjects/SeeSR/preset/models/prompt_gan/select/generator_epoch_100.pth')
    parser.add_argument("--vocabulary_path", type=str, default="F:/PyCharmProjects/SeeSR/lexicon.json",
                        help="The path to the vocabulary JSON file")
    args = parser.parse_args()
    main(args)



