import json
from nltk.corpus import wordnet as wn
import random

# Ensure NLTK wordnet data is available; if not, download it
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')

# Helper to get adjectives and nouns from WordNet
def get_words(pos, limit):
    words = set()
    for syn in wn.all_synsets(pos=pos):
        for lemma in syn.lemmas():
            word = lemma.name().replace('_', ' ')
            if word.isalpha() and len(word) > 1:
                words.add(word.lower())
            if len(words) >= limit:
                break
        if len(words) >= limit:
            break
    return list(words)

# Generate ~3000 adjectives and nouns
adjectives = get_words('a', 3000)
nouns = get_words('n', 3000)

# Define some subcategories heuristically
colors = ['red', 'blue', 'green', 'yellow', 'black', 'white', 'orange', 'purple', 'pink', 'brown', 'gray']
shapes = ['circular', 'square', 'triangular', 'rectangular', 'oval', 'hexagonal']
textures = adjectives[:50]  # for demo, first 50 adjectives as textures

# Build JSON structure
lexicon = {
    "adjective": {
        "colors": colors,
        "textures": textures,
        "shapes": shapes,
        "others": [adj for adj in adjectives if adj not in colors + textures + shapes][:3000 - (len(colors) + len(textures) + len(shapes))]
    },
    "nouns": {
        "natural": nouns[:1500],
        "artificial": nouns[1500:3000]
    },
    "combination_rules": [
        "adjective + noun",
        "adjective + adjective + noun"
    ]
}

# Save to file
from pathlib import Path

#Obtain the directory where the current script is located
base_dir = Path(__file__).resolve().parent
out_path = base_dir / 'lexicon.json'
with out_path.open('w', encoding='utf-8') as f:
    json.dump(lexicon, f, ensure_ascii=False, indent=2)
print(f"Saved lexicon to {out_path}")

# Display a preview
import pandas as pd
preview = pd.DataFrame({
    'adjective_colors': colors,
    'adjective_shapes': shapes + ['']*(len(colors)-len(shapes)),
    'noun_natural': lexicon['nouns']['natural'][:len(colors)],
    'noun_artificial': lexicon['nouns']['artificial'][:len(colors)]
})
import ace_tools as tools
tools.display_dataframe_to_user("Lexicon Preview", preview)
