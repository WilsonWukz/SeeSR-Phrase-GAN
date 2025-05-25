import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from Prompt_Dataset import PromptDataset
from prompt_gan import PromptGenerator, PromptDiscriminator
import clip
import pickle
from tqdm import tqdm


def train_prompt_gan(imgid_to_vector, vocab_words, image_dir):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # CLIP for text alignment only
    clip_model, preprocess = clip.load("ViT-B/32", device=device)

    # annotation_csv = "E:/Sydney_study/5329/A2/flickr/flickr_annotations_30k.csv"

    dataset = PromptDataset(
        image_dir=image_dir,
        imgid_to_vector=imgid_to_vector,
        vocab_words=vocab_words,
        imageid_to_filename=imageid_to_filename
    )
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    output_dim = len(vocab_words)

    G = PromptGenerator(output_dim=output_dim).to(device)   # 用 U-Net 替代 MLP
    D = PromptDiscriminator(input_dim=output_dim).to(device)
    print(output_dim)
    print("Generator:", G)
    print("Discriminator:", D)

    opt_G = torch.optim.Adam(G.parameters(), lr=5e-4)
    opt_D = torch.optim.Adam(D.parameters(), lr=1e-4)

    for epoch in tqdm(range(25)):
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)

            # ========== Train D ==========
            with torch.no_grad():
                fake = torch.sigmoid(G(imgs)).detach()  # 🔁 输入图像（不是 CLIP 特征）

            real_score = D(labels)
            fake_score = D(fake)
            loss_D = -torch.mean(torch.log(real_score + 1e-5) + torch.log(1 - fake_score + 1e-5))
            opt_D.zero_grad()
            loss_D.backward()
            opt_D.step()

            # ========== Train G ==========
            pred = torch.sigmoid(G(imgs))  # 🔁 输入图像（不是 CLIP 特征）
            fake_score = D(pred)
            loss_G_adv = -torch.mean(torch.log(fake_score + 1e-5))

            norm_pred = F.normalize(pred, dim=1)  # 对每个向量进行归一化 [B, output_dim]
            sim_matrix = torch.matmul(norm_pred, norm_pred.T)  # [B, B]，每对样本之间的余弦相似度
            batch_size = pred.size(0)
            mask = torch.eye(batch_size, device=pred.device).bool()
            sim_matrix.masked_fill_(mask, 0)  # 不考虑对角线（自己和自己）
            diversity_loss = sim_matrix.mean()  # 越小越好，表示越“多样”

            # 加权加入到 Generator 总 loss
            λ_div = 0.1  # 控制多样性正则项权重

            # CLIP similarity loss
            captions = []
            for p in pred.cpu():
                indices = (p > 0.9).nonzero().flatten().tolist()
                if len(indices) > 10:
                    indices = indices[:10]  # 防止过长
                caption = " ".join([vocab_words[i] for i in indices])
                if len(caption.strip()) == 0:
                    caption = "realistic"
                captions.append(caption)

            text_inputs = clip.tokenize(captions).to(device)
            with torch.no_grad():
                text_feat = clip_model.encode_text(text_inputs)
                img_feat = clip_model.encode_image(imgs).float()  # 🔁 用于计算语义对齐 loss

            sim = F.cosine_similarity(img_feat, text_feat).mean()
            loss_clip = 1 - sim

            loss_G = loss_G_adv + 0.5 * loss_clip + λ_div * diversity_loss
            opt_G.zero_grad()
            loss_G.backward()
            opt_G.step()
            # print(captions)
        print(f"Epoch {epoch}: D={loss_D.item():.4f}, G={loss_G.item():.4f}, Div={diversity_loss.item():.4f}")

    torch.save(G.state_dict(), "E:/Sydney_study/5329/A2/flickr/model/coco_generator2.pth")


# ====== 数据加载入口点 ======
with open("E:/Sydney_study/5329/A2/COCO/imgid_to_vector.pkl", "rb") as f:
    imgid_to_vector = pickle.load(f)

with open("E:/Sydney_study/5329/A2/COCO/vocab_words.txt", "r", encoding="utf-8") as f:
    vocab_words = [line.strip() for line in f.readlines()]

with open("E:/Sydney_study/5329/A2/COCO/image_id_to_filename.pkl", "rb") as f:
    imageid_to_filename = pickle.load(f)

train_prompt_gan(imgid_to_vector, vocab_words, "E:/Sydney_study/5329/A2/COCO/data/val2014/")
