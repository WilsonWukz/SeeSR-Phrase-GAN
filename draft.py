# test_phrase_generator.py
from utils_text.phrase_generator import PhraseGenerator

#Initialize the phrase generator
generator = PhraseGenerator("F:/PyCharmProjects/SeeSR/lexicon.json")

#Test some nouns
test_nouns = ["animal", "salamander", "stone", "pink"]
for noun in test_nouns:
    phrase = generator.generate_phrase_for_noun(noun)
    print(f"{noun} -> {phrase}")

#Test enhance_prompt
tags = ["animal", "pink", "stone", "salamander"]
enhanced = generator.enhance_prompt(tags)
print(f"Enhanced: {enhanced}")
