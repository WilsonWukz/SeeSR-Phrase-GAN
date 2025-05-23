import torch
from PIL import Image
from prompt_gan import PromptGenerator
import clip


def load_vocab(vocab_path):
    with open(vocab_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f.readlines()]


def load_generator(model_path, output_dim, device):
    model = PromptGenerator(output_dim=output_dim).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def predict_adjectives(image, generator, vocab_words, clip_model, preprocess):
    device = next(generator.parameters()).device
    img_tensor = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        img_feat = clip_model.encode_image(img_tensor).float()
        logits = generator(img_feat)
        probs = torch.sigmoid(logits).squeeze()

        indices = (probs > 0.5).nonzero(as_tuple=True)[0]
        adjectives = [vocab_words[i] for i in indices.tolist()]
        return adjectives



if __name__ == "__main__":
    # ===== 路径设置 =====
    vocab_path = "E:/Sydney_study/5329/A2/flickr/vocab_words.txt"
    model_path = "E:/Sydney_study/5329/A2/flickr/model/generator.pth"
    image_path = "E:/Sydney_study/5329/A2/flickr/data/flickr30k-images/1000092795.jpg"

    # ===== 加载组件 =====
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, preprocess = clip.load("ViT-B/32", device=device)
    vocab_words = load_vocab(vocab_path)
    generator = load_generator(model_path, output_dim=len(vocab_words), device=device)

    # ===== 推理 =====
    adjectives = predict_adjectives(image_path, generator, vocab_words, clip_model, preprocess)
    print(len(adjectives))
    print("预测形容词列表:", adjectives)
