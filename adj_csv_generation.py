from transformers import pipeline
import json
import pandas as pd
from tqdm import tqdm

with open("/content/captions_val2014.json", "r", encoding="utf-8") as f:
    coco_data = json.load(f)

#Extract image_id → captions
annotations = coco_data["annotations"]
image_id_to_captions = {}
for item in annotations:
    image_id = item["image_id"]
    caption = item["caption"]
    image_id_to_captions.setdefault(image_id, []).append(caption)

#Load the POS annotator (BERT-based)
pos_tagger = pipeline("token-classification", model="vblagoje/bert-english-uncased-finetuned-pos", aggregation_strategy="simple")

#Extract the ADJ + NOUN pairing
adj_noun_pairs = []
for image_id, captions in tqdm(image_id_to_captions.items()):
    for caption in captions:
        try:
            results = pos_tagger(caption.lower())
        except:
            continue
        words = [r["word"] for r in results]
        tags = [r["entity_group"] for r in results]
        for i in range(len(tags) - 1):
            if tags[i] == "ADJ" and tags[i + 1] == "NOUN":
                adj_noun_pairs.append({
                    "image_id": image_id,
                    "adjective": words[i],
                    "noun": words[i + 1]
                })

df = pd.DataFrame(adj_noun_pairs)
df.to_csv("/content/adj_noun_pairs_val2014.csv", index=False)
