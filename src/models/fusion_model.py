from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class FusionModelConfig:
    embed_dim: int = 768
    num_heads: int = 8
    num_layers: int = 3
    audio_dim: int = 40
    face_dim: int = 192
    target_length: int = 512
    dropout: float = 0.5


class MultiModalCrossAttention(nn.Module):
    """PCLMM fusion model compatible with the official checkpoint.

    The module names intentionally match `cross_attention_without_xml.py`,
    so a state_dict saved by the official script can be loaded directly.
    """

    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 8,
        num_layers: int = 3,
        audio_dim: int = 40,
        face_dim: int = 192,
        target_length: int = 512,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.audio_fc = nn.Linear(audio_dim, embed_dim)
        self.face_fc = nn.Linear(face_dim * 4, embed_dim)

        self.text_transformer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, batch_first=True
        )
        self.audio_transformer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, batch_first=True
        )
        self.video_transformer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, batch_first=True
        )
        self.face_transformer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, batch_first=True
        )

        self.cross_attention_text_audio = nn.MultiheadAttention(
            embed_dim, num_heads, batch_first=True
        )
        self.cross_attention_text_video = nn.MultiheadAttention(
            embed_dim, num_heads, batch_first=True
        )
        self.cross_attention_text_face = nn.MultiheadAttention(
            embed_dim, num_heads, batch_first=True
        )
        self.cross_attention_audio_video = nn.MultiheadAttention(
            embed_dim, num_heads, batch_first=True
        )
        self.cross_attention_audio_face = nn.MultiheadAttention(
            embed_dim, num_heads, batch_first=True
        )
        self.cross_attention_video_face = nn.MultiheadAttention(
            embed_dim, num_heads, batch_first=True
        )

        self.num_layers = num_layers
        self.target_length = target_length
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(embed_dim, 1)

    def forward(
        self,
        text_out: torch.Tensor,
        audio_out: torch.Tensor,
        video_out: torch.Tensor,
        face_out: torch.Tensor,
    ) -> torch.Tensor:
        audio_out = self.audio_fc(audio_out)

        face_out = self._flatten_face(face_out)
        face_out = self.face_fc(face_out.float())

        if text_out.dim() == 2:
            text_out = text_out.unsqueeze(1)
        if audio_out.dim() == 2:
            audio_out = audio_out.unsqueeze(1)
        if video_out.dim() == 2:
            video_out = video_out.unsqueeze(1)
        if face_out.dim() == 2:
            face_out = face_out.unsqueeze(1)

        text_out = self._pool_to_target_length(text_out)
        audio_out = self._pool_to_target_length(audio_out)
        video_out = self._pool_to_target_length(video_out)
        face_out = self._pool_to_target_length(face_out)

        for _ in range(self.num_layers):
            text_out = self.text_transformer(text_out)
            audio_out = self.audio_transformer(audio_out)
            video_out = self.video_transformer(video_out)
            face_out = self.face_transformer(face_out)

            text_audio_attn, _ = self.cross_attention_text_audio(
                text_out, audio_out, audio_out
            )
            text_video_attn, _ = self.cross_attention_text_video(
                text_out, video_out, video_out
            )
            text_face_attn, _ = self.cross_attention_text_face(
                text_out, face_out, face_out
            )
            audio_video_attn, _ = self.cross_attention_audio_video(
                audio_out, video_out, video_out
            )
            audio_face_attn, _ = self.cross_attention_audio_face(
                audio_out, face_out, face_out
            )
            video_face_attn, _ = self.cross_attention_video_face(
                video_out, face_out, face_out
            )

            text_out = text_out + text_audio_attn + text_video_attn + text_face_attn
            audio_out = audio_out + text_audio_attn + audio_video_attn + audio_face_attn
            video_out = video_out + text_video_attn + audio_video_attn + video_face_attn
            face_out = face_out + text_face_attn + audio_face_attn + video_face_attn

            text_out = self.dropout(text_out)
            audio_out = self.dropout(audio_out)
            video_out = self.dropout(video_out)
            face_out = self.dropout(face_out)

        combined_out = text_out + audio_out + video_out + face_out
        combined_out = torch.mean(combined_out, dim=1)
        return self.fc(combined_out)

    def _pool_to_target_length(self, tensor: torch.Tensor) -> torch.Tensor:
        return F.adaptive_avg_pool1d(
            tensor.transpose(1, 2), self.target_length
        ).transpose(1, 2)

    @staticmethod
    def _flatten_face(face_out: torch.Tensor) -> torch.Tensor:
        if face_out.dim() == 5:
            batch_size, seq_len = face_out.shape[:2]
            return face_out.view(batch_size, seq_len, -1)
        if face_out.dim() == 4:
            batch_size, seq_len = face_out.shape[:2]
            return face_out.view(batch_size, seq_len, -1)
        if face_out.dim() == 3 and face_out.shape[-1] == 192:
            return face_out.repeat_interleave(4, dim=-1)
        raise ValueError(f"Unsupported face feature shape: {tuple(face_out.shape)}")


def build_fusion_model(config: FusionModelConfig | None = None) -> MultiModalCrossAttention:
    config = config or FusionModelConfig()
    return MultiModalCrossAttention(
        embed_dim=config.embed_dim,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        audio_dim=config.audio_dim,
        face_dim=config.face_dim,
        target_length=config.target_length,
        dropout=config.dropout,
    )


def load_checkpoint_state(checkpoint_path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        checkpoint = checkpoint["model_state_dict"]

    if not isinstance(checkpoint, dict):
        raise TypeError(f"Unsupported checkpoint type: {type(checkpoint)!r}")

    state = {}
    for key, value in checkpoint.items():
        if key.startswith("module."):
            key = key[len("module.") :]
        state[key] = value
    return state


def load_fusion_model(
    checkpoint_path: str | Path,
    config: FusionModelConfig | None = None,
    device: str | torch.device = "cpu",
    strict: bool = True,
) -> MultiModalCrossAttention:
    model = build_fusion_model(config)
    state = load_checkpoint_state(checkpoint_path, map_location="cpu")
    model.load_state_dict(state, strict=strict)
    model.to(device)
    model.eval()
    return model
