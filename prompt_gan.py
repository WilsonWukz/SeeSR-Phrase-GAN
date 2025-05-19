import torch
import torch.nn as nn


class PromptEnhancementGenerator(nn.Module):
    def __init__(self, tag_dim=768, image_embed_dim=512, hidden_dim=1024):
        super().__init__()
        self.tag_encoder = nn.Linear(tag_dim, hidden_dim)
        self.image_encoder = nn.Linear(image_embed_dim, hidden_dim)

        self.layers = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.GELU(),
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, tag_dim)
        )

    def forward(self, tag_embeds, image_embeds):
        tag_features = self.tag_encoder(tag_embeds)
        image_features = self.image_encoder(image_embeds)
        combined = torch.cat([tag_features, image_features], dim=-1)
        enhanced_tag_embeds = self.layers(combined)
        return enhanced_tag_embeds


class PromptEnhancementDiscriminator(nn.Module):
    def __init__(self, tag_dim=768, image_embed_dim=512, hidden_dim=1024):
        super().__init__()
        self.tag_encoder = nn.Linear(tag_dim, hidden_dim)
        self.image_encoder = nn.Linear(image_embed_dim, hidden_dim)

        self.layers = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.LayerNorm(hidden_dim // 2),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, tag_embeds, image_embeds):
        if image_embeds.dim() == 4:
            image_embeds = torch.nn.functional.adaptive_avg_pool2d(image_embeds, 1)
            image_embeds = image_embeds.view(image_embeds.size(0), -1)

        tag_features = self.tag_encoder(tag_embeds)
        image_features = self.image_encoder(image_embeds)
        combined = torch.cat([tag_features, image_features], dim=-1)
        return self.layers(combined)