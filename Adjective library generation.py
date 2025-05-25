from transformers import pipeline
import pandas as pd
from collections import defaultdict
import pickle
import json
from tqdm import tqdm

# 1. 加载 BERT POS Tagger
pos_tagger = pipeline("token-classification",
                      model="vblagoje/bert-english-uncased-finetuned-pos",
                      aggregation_strategy="simple")

# 2. 加载形容词词汇表（你刚提取的）
vocab_df = pd.read_csv("E:/Sydney_study/5329/A2/COCO/coco_val2014_adjectives.csv")
vocab_words = list(vocab_df["Adjective"])
word2id = {w: i for i, w in enumerate(vocab_words)}

# 3. 加载 COCO captions JSON
with open("E:/Sydney_study/5329/A2/COCO/annotations/captions_val2014.json", "r", encoding="utf-8") as f:
    coco_data = json.load(f)

# 4. 构建 image_id → file_name 映射
image_id_to_filename = {
    img["id"]: img["file_name"] for img in coco_data["images"]
}

# 5. 遍历每条 caption，收集形容词
imgid_to_adjs = defaultdict(set)

for ann in tqdm(coco_data["annotations"]):
    image_id = ann["image_id"]
    caption = ann["caption"].lower()
    results = pos_tagger(caption)

    for item in results:
        word = item["word"]
        pos = item["entity_group"]
        if pos == "ADJ" and word in word2id:
            imgid_to_adjs[image_id].add(word)

# 6. 构建标签向量
imgid_to_vector = {}
for img_id, adjs in imgid_to_adjs.items():
    vec = [0] * len(vocab_words)
    for adj in adjs:
        vec[word2id[adj]] = 1
    imgid_to_vector[img_id] = vec

# 7. 保存为 pickle + vocab txt
with open("E:/Sydney_study/5329/A2/COCO/imgid_to_vector.pkl", "wb") as f:
    pickle.dump(imgid_to_vector, f)

with open("E:/Sydney_study/5329/A2/COCO/vocab_words.txt", "w", encoding="utf-8") as f:
    for word in vocab_words:
        f.write(word + "\n")
# 8. 可选：保存 image_id → filename 映射
with open("E:/Sydney_study/5329/A2/COCO/image_id_to_filename.pkl", "wb") as f:
    pickle.dump(image_id_to_filename, f)

print("✅ COCO 标签映射 & 文件名映射生成完成。")
