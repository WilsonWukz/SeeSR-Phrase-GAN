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

    annotation_csv = "E:/Sydney_study/5329/A2/flickr/flickr_annotations_30k.csv"

    dataset = PromptDataset(
        image_dir=image_dir,
        imgid_to_vector=imgid_to_vector,
        vocab_words=vocab_words,
        annotation_csv=annotation_csv
    )
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    output_dim = len(vocab_words)
    G = PromptGenerator(output_dim=output_dim).to(device)   # 用 U-Net 替代 MLP
    D = PromptDiscriminator(input_dim=output_dim).to(device)

    print("Generator:", G)
    print("Discriminator:", D)

    opt_G = torch.optim.Adam(G.parameters(), lr=1e-4)
    opt_D = torch.optim.Adam(D.parameters(), lr=1e-4)

    for epoch in tqdm(range(20)):
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

            # CLIP similarity loss
            captions = [" ".join([vocab_words[i] for i in (p > 0.5).nonzero().flatten().tolist()]) for p in pred.cpu()]
            text_inputs = clip.tokenize(captions).to(device)
            with torch.no_grad():
                text_feat = clip_model.encode_text(text_inputs)
                img_feat = clip_model.encode_image(imgs).float()  # 🔁 用于计算语义对齐 loss

            sim = F.cosine_similarity(img_feat, text_feat).mean()
            loss_clip = 1 - sim

            loss_G = loss_G_adv + 0.5 * loss_clip
            opt_G.zero_grad()
            loss_G.backward()
            opt_G.step()

        print(f"Epoch {epoch}: D={loss_D.item():.4f}, G={loss_G.item():.4f}")

    torch.save(G.state_dict(), "E:/Sydney_study/5329/A2/flickr/model/generator.pth")


# ====== 数据加载入口点 ======
with open("E:/Sydney_study/5329/A2/flickr/imgid_to_vector.pkl", "rb") as f:
    imgid_to_vector = pickle.load(f)

with open("E:/Sydney_study/5329/A2/flickr/vocab_words.txt", "r", encoding="utf-8") as f:
    vocab_words = [line.strip() for line in f.readlines()]

train_prompt_gan(imgid_to_vector, vocab_words, "E:/Sydney_study/5329/A2/flickr/data/flickr30k-images")
