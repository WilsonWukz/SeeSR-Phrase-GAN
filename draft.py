# test_phrase_generator.py
from utils_text.phrase_generator import PhraseGenerator

# 初始化短语生成器
generator = PhraseGenerator("F:/PyCharmProjects/SeeSR/lexicon.json")

# 测试一些名词
test_nouns = ["animal", "salamander", "stone", "pink"]
for noun in test_nouns:
    phrase = generator.generate_phrase_for_noun(noun)
    print(f"{noun} -> {phrase}")

# 测试enhance_prompt
tags = ["animal", "pink", "stone", "salamander"]
enhanced = generator.enhance_prompt(tags)
print(f"Enhanced: {enhanced}")