import json
import random
import os
from typing import List, Dict, Any, Optional


class PhraseGenerator:
    def __init__(self, vocabulary_path: str):
        """
        Initialize the phrase generator

        Args:
            vocabulary_path: the path of the vocabulary file
        """
        self.vocabulary = self._load_vocabulary(vocabulary_path)

    def _load_vocabulary(self, path: str) -> Dict[str, Any]:
        """Load the vocabulary file """
        if not os.path.exists(path):
            raise FileNotFoundError(f"The vocabulary file doesn't exist: {path}")

        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def generate_phrase_for_noun(self, noun: str) -> str:
        """
        Generate a phrase for noun

        Args:
            noun

        Returns:
            The format is: "adjective + noun"
        """
        noun_category = self._get_noun_category(noun)

        if not noun_category:
            # 如果找不到名词类别，随机选择一个定语
            adjectives = []
            if "adjective" in self.vocabulary:
                for adj_category in self.vocabulary["adjective"]:
                    adjectives.extend(self.vocabulary["adjective"][adj_category])

            if not adjectives:
                return noun

            adjective = random.choice(adjectives)
            return f"{adjective} {noun}"

        suitable_adj_categories = self._get_suitable_adjective_categories(noun_category)

        adjective = self._select_adjective(suitable_adj_categories)

        return f"{adjective} {noun}"

    def _get_noun_category(self, noun: str) -> Optional[str]:
        """Determine the category to which a noun belongs"""
        if "nouns" not in self.vocabulary:
            return None

        for category, nouns in self.vocabulary["nouns"].items():
            if noun in nouns:
                return category

        return None

    def _get_suitable_adjective_categories(self, noun_category: str) -> List[str]:
        """根据名词类别获取合适的定语类别"""
        if "combination_rules" in self.vocabulary:
            for rule in self.vocabulary["combination_rules"]:
                if isinstance(rule, dict) and rule.get("noun_category") == noun_category:
                    return rule.get("adjective_categories", [])

                    # 如果没有找到匹配的规则，返回所有定语类别
        if "adjective" in self.vocabulary:
            return list(self.vocabulary["adjective"].keys())
        return []

    def _select_adjective(self, categories: List[str]) -> str:
        """从给定的类别中选择一个定语"""
        # 首先尝试从指定类别中选择
        adjectives = []
        for category in categories:
            if "adjective" in self.vocabulary and category in self.vocabulary["adjective"]:
                adjectives.extend(self.vocabulary["adjective"][category])

                # 如果没有找到合适的定语，从所有定语中选择
        if not adjectives:
            if "adjective" in self.vocabulary:
                for cat in self.vocabulary["adjective"]:
                    adjectives.extend(self.vocabulary["adjective"][cat])

        if not adjectives:
            return ""  # 如果没有定语可用，返回空字符串

        return random.choice(adjectives)

    def enhance_prompt(self, tags: List[str]) -> str:
        """
        增强提示词，为标签添加定语

        Args:
            tags: 图像标签列表

        Returns:
            增强后的提示词
        """
        enhanced_tags = []

        for tag in tags:
            # 简单处理：将标签分割成单词，假设最后一个单词是名词
            words = tag.strip().split()

            # 如果标签已经包含多个单词，可能已经有定语了
            if len(words) > 1:
                enhanced_tags.append(tag)
                continue

            noun = words[0]
            enhanced_tag = self.generate_phrase_for_noun(noun)
            enhanced_tags.append(enhanced_tag)

        return ", ".join(enhanced_tags)