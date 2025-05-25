from transformers import pipeline
import pandas as pd
from collections import defaultdict
import pickle
import json
from tqdm import tqdm

#Load BERT POS Tagger
pos_tagger = pipeline("token-classification",
                      model="vblagoje/bert-english-uncased-finetuned-pos",
                      aggregation_strategy="simple")

#Load the adjective vocabulary
vocab_df = pd.read_csv("E:/Sydney_study/5329/A2/COCO/coco_val2014_adjectives.csv")
vocab_words = list(vocab_df["Adjective"])
word2id = {w: i for i, w in enumerate(vocab_words)}

#Load the COCO captions JSON
with open("E:/Sydney_study/5329/A2/COCO/annotations/captions_val2014.json", "r", encoding="utf-8") as f:
    coco_data = json.load(f)

#Construct the image_id → file_name mapping
image_id_to_filename = {
    img["id"]: img["file_name"] for img in coco_data["images"]
}

#Traverse each caption and collect adjectives
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

#Construct the label vector
imgid_to_vector = {}
for img_id, adjs in imgid_to_adjs.items():
    vec = [0] * len(vocab_words)
    for adj in adjs:
        vec[word2id[adj]] = 1
    imgid_to_vector[img_id] = vec

#Save as pickle + vocab txt
with open("E:/Sydney_study/5329/A2/COCO/imgid_to_vector.pkl", "wb") as f:
    pickle.dump(imgid_to_vector, f)

with open("E:/Sydney_study/5329/A2/COCO/vocab_words.txt", "w", encoding="utf-8") as f:
    for word in vocab_words:
        f.write(word + "\n")
with open("E:/Sydney_study/5329/A2/COCO/image_id_to_filename.pkl", "wb") as f:
    pickle.dump(image_id_to_filename, f)

print("The COCO label mapping and file name mapping have been generated")
