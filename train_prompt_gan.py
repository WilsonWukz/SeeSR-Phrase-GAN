import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPTokenizer, CLIPTextModel
from models.prompt_gan import PromptEnhancementGenerator, PromptEnhancementDiscriminator
from ram.models.ram_lora import ram
from ram import inference_ram as inference
from torchvision import transforms
import torch.nn.functional as F
from PIL import Image
import os, json, random
import pandas as pd
from torch.utils.data import Dataset
from transformers import CLIPTokenizer, CLIPTextModel
from ram.models.ram_lora import ram
from ram import inference_ram as inference
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

class PromptEnhancementDataset(Dataset):
    def __init__(self, image_dir, csv_file, pretrained_model_path, ram_model_path, device='cuda'):
        self.image_dir = image_dir
        self.device = device

        # 读取 CSV 并解析 raw 列
        df = pd.read_csv(csv_file)
        df['raw_list'] = df['raw'].apply(json.loads)
        # 可选：根据 split 列划分数据集
        # df = df[df['split'] == 'train']

        # 构建样本列表
        self.samples = []
        for _, row in df.iterrows():
            img_path = os.path.join(image_dir, row['filename'])
            if not os.path.isfile(img_path):
                raise FileNotFoundError(f"Missing image file: {img_path}")
            self.samples.append((img_path, row['raw_list']))

        # 加载 RAM 模型
        self.ram_model = ram(
            pretrained='preset/models/ram_swin_large_14m.pth',
            pretrained_condition=ram_model_path,
            image_size=384,
            vit='swin_l'
        )
        self.ram_model.eval().to(device)

        # 加载 CLIP 文本编码器
        self.tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(pretrained_model_path, subfolder="text_encoder")
        self.text_encoder.eval().to(device)

        # 图像预处理
        self.to_tensor = transforms.ToTensor()
        self.ram_transforms = transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, raw_prompts = self.samples[idx]

        # 图像 → RAM 嵌入
        image = Image.open(img_path).convert("RGB")
        lq = self.to_tensor(image).unsqueeze(0).to(self.device)
        lq = self.ram_transforms(lq)
        raw_feats = self.ram_model.generate_image_embeds(lq)
        if raw_feats.dim() == 4:
            pooled = F.adaptive_avg_pool2d(raw_feats, 1).view(1, -1)
            ram_states = pooled.squeeze(0)
        else:
            ram_states = raw_feats.mean(dim=1).squeeze(0)

        # RAM prompt → 粗标签嵌入
        res = inference(lq, self.ram_model)
        inputs = self.tokenizer(
            res[0], padding="max_length", truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt"
        ).to(self.device)
        tag_embeds = self.text_encoder(**inputs).last_hidden_state.mean(dim=1).squeeze(0)

        # 随机选一条高质量提示进行训练
        sent = random.choice(raw_prompts)
        toks = self.tokenizer(
            sent, padding="max_length", truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt"
        ).to(self.device)
        target_embeds = self.text_encoder(**toks).last_hidden_state.mean(dim=1).squeeze(0)

        return {
            "tag_embeds": tag_embeds,                  # [768]
            "ram_encoder_hidden_states": ram_states,   # [D]
            "target_embeds": target_embeds             # [768]
        }


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 创建数据集和数据加载器
    dataset = PromptEnhancementDataset(
        args.image_dir,
        args.prompts_file,
        args.pretrained_model_path,
        args.ram_model_path,
        device
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # 初始化生成器和判别器
    text_embed_dim = dataset.text_encoder.config.hidden_size
    generator = PromptEnhancementGenerator(tag_dim=text_embed_dim, image_embed_dim=512, hidden_dim=1024)
    discriminator = PromptEnhancementDiscriminator(tag_dim=text_embed_dim, image_embed_dim=512, hidden_dim=1024)

    generator.to(device)
    discriminator.to(device)

    # 定义优化器
    g_optimizer = optim.Adam(generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    d_optimizer = optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))

    # 定义损失函数
    adversarial_loss = nn.BCEWithLogitsLoss()
    content_loss = nn.MSELoss()

    # 训练循环
    for epoch in range(args.epochs):
        for i, batch in enumerate(dataloader):
            tag_embeds = batch["tag_embeds"]
            ram_encoder_hidden_states = batch["ram_encoder_hidden_states"]
            target_embeds = batch["target_embeds"]

            target_embeds = batch["target_embeds"]

            print(f"[DEBUG] tag_embeds.shape = {tag_embeds.shape}")
            print(f"[DEBUG] ram_encoder_hidden_states.shape = {ram_encoder_hidden_states.shape}")

            # 创建真假标签
            valid = torch.ones(tag_embeds.size(0), 1, device=device)
            fake = torch.zeros(tag_embeds.size(0), 1, device=device)

            # ---------------------
            #  训练生成器
            # ---------------------
            g_optimizer.zero_grad()
            # 冻结判别器

            for p in discriminator.parameters():
                p.requires_grad = False
            g_optimizer.zero_grad()
            # 生成增强的标签嵌入
            enhanced_tag_embeds = generator(tag_embeds, ram_encoder_hidden_states)

            # 计算内容损失（与目标高质量嵌入的差异）
            c_loss = content_loss(enhanced_tag_embeds, target_embeds)

            # 判别器对生成的嵌入的评估
            validity = discriminator(enhanced_tag_embeds, ram_encoder_hidden_states)

            # 对抗损失
            g_loss = adversarial_loss(validity, valid) + args.lambda_content * c_loss

            g_loss.backward(retain_graph=True)
            g_optimizer.step()

            for p in discriminator.parameters():
                p.requires_grad = True

            # ---------------------
            #  训练判别器
            # ---------------------
            if i % args.n_critic == 0:
                d_optimizer.zero_grad()

                # 判别器对真实高质量嵌入的评估
                real_validity = discriminator(target_embeds, ram_encoder_hidden_states)
                real_loss = adversarial_loss(real_validity, valid)

                # 判别器对生成的嵌入的评估
                fake_validity = discriminator(enhanced_tag_embeds.detach(), ram_encoder_hidden_states)
                fake_loss = adversarial_loss(fake_validity, fake)

                # 总判别器损失
                d_loss = (real_loss + fake_loss) / 2

                d_loss.backward()
                d_optimizer.step()

                # 打印训练进度
            if i % args.print_freq == 0:
                print(
                    f"[Epoch {epoch}/{args.epochs}] [Batch {i}/{len(dataloader)}] "
                    f"[D loss: {d_loss.item():.4f}] [G loss: {g_loss.item():.4f} (Adv: {adversarial_loss(validity, valid).item():.4f}, Content: {c_loss.item():.4f})]"
                )

                # 每个epoch保存一次模型
        if (epoch + 1) % 1 == 0:
            os.makedirs(args.output_dir, exist_ok=True)
            torch.save(generator.state_dict(), f"{args.output_dir}/generator_epoch_{epoch + 1}.pth")
            torch.save(discriminator.state_dict(), f"{args.output_dir}/discriminator_epoch_{epoch + 1}.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=str, required=False, default="F:/PyCharmProjects/SeeSR/GAN/img/train2014/flickr30k-images")
    parser.add_argument("--prompts_file", type=str, required=False, default="F:/PyCharmProjects/SeeSR/GAN/txt/flickr_annotations_10k.csv")
    parser.add_argument("--pretrained_model_path", type=str, default="preset/models/stable-diffusion-2-base",
                        help="pre-trained model path")
    parser.add_argument("--ram_model_path", type=str, default="preset/models/DAPE.pth", help="RAM model path")
    parser.add_argument("--output_dir", type=str, default="preset/models/prompt_gan", help="output directory")
    parser.add_argument("--batch_size", type=int, default=4, help="batch size")
    parser.add_argument("--lr", type=float, default=0.0002, help="learning rate")
    parser.add_argument("--epochs", type=int, default=100, help="number of training rounds")
    parser.add_argument("--n_critic", type=int, default=5, help="discriminator training frequency")
    parser.add_argument("--lambda_content", type=float, default=10.0, help="content loss weight")
    parser.add_argument("--print_freq", type=int, default=10, help="print frequency")
    parser.add_argument("--save_freq", type=int, default=5, help="save frequency")

    args = parser.parse_args()
    train(args)