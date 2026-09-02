"""BreakTypeNet model definition."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LiteViT(nn.Module):
    """Shared lightweight frame encoder."""

    def __init__(
        self,
        img_size=224,
        patch_size=14,
        in_chans=3,
        embed_dim=384,
        depth=6,
        num_heads=6,
        mlp_ratio=4.0,
    ):
        super().__init__()
        self.patch_embed = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        num_patches = (img_size // patch_size) ** 2
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=int(embed_dim * mlp_ratio),
                dropout=0.1,
                batch_first=True,
            )
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

    def forward(self, frames):
        tokens = self.patch_embed(frames).flatten(2).transpose(1, 2)
        cls_tokens = self.cls_token.expand(tokens.shape[0], -1, -1)
        tokens = torch.cat((cls_tokens, tokens), dim=1) + self.pos_embed
        for block in self.blocks:
            tokens = block(tokens)
        return self.norm(tokens)[:, 1:]


class LSTMClassifier(nn.Module):
    """Three-layer unidirectional temporal classifier."""

    def __init__(
        self,
        input_dim=384,
        hidden_dim=256,
        layers=3,
        fc_dim=128,
        dropout=0.3,
        num_classes=3,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=layers,
            batch_first=True,
        )
        self.fc1 = nn.Linear(hidden_dim, fc_dim)
        self.fc2 = nn.Linear(fc_dim, num_classes)
        self.dropout = dropout

    def forward(self, frame_features):
        self.lstm.flatten_parameters()
        sequence, _ = self.lstm(frame_features)
        output = F.relu(self.fc1(sequence[:, -1]))
        output = F.dropout(output, p=self.dropout, training=self.training)
        return self.fc2(output)


class FeatureAdapter(nn.Module):
    """Map 384-dimensional student patches to the 768-dimensional teacher space."""

    def __init__(self, student_dim=384, teacher_dim=768):
        super().__init__()
        self.adapter = nn.Linear(student_dim, teacher_dim)

    def forward(self, features):
        return self.adapter(features)


class BreakTypeNet(nn.Module):
    """LiteViT frame encoding followed by LSTM video classification."""

    def __init__(self, num_classes=3, embed_dim=384):
        super().__init__()
        self.vit = LiteViT(embed_dim=embed_dim)
        self.lstm = LSTMClassifier(input_dim=embed_dim, num_classes=num_classes)

    def forward(self, videos):
        batch_size, sequence_length, channels, height, width = videos.shape
        frames = videos.reshape(
            batch_size * sequence_length, channels, height, width
        )
        patches = self.vit(frames)
        num_patches, feature_dim = patches.shape[1:]
        patches = patches.reshape(
            batch_size, sequence_length, num_patches, feature_dim
        )
        frame_features = patches.mean(dim=2)
        logits = self.lstm(frame_features)
        distillation_features = patches.mean(dim=1)
        return logits, distillation_features
