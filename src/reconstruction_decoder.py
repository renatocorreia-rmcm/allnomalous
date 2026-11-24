"""Learned-query decoder for frozen spatial feature reconstruction."""

from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn


class _EncoderBlock(nn.Module):
    def __init__(
        self, hidden_dim: int, num_heads: int, feedforward_dim: int, dropout: float
    ):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.linear1 = nn.Linear(hidden_dim, feedforward_dim)
        self.linear2 = nn.Linear(feedforward_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout_attention = nn.Dropout(dropout)
        self.dropout_ffn = nn.Dropout(dropout)
        self.dropout_output = nn.Dropout(dropout)

    def forward(
        self,
        source: torch.Tensor,
        position: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        query_key = source + position
        attended = self.attention(
            query_key, query_key, source, attn_mask=attention_mask, need_weights=False
        )[0]
        source = self.norm1(source + self.dropout_attention(attended))
        feedforward = self.linear2(self.dropout_ffn(F.gelu(self.linear1(source))))
        return self.norm2(source + self.dropout_output(feedforward))


class _DecoderBlock(nn.Module):
    def __init__(
        self, hidden_dim: int, num_heads: int, feedforward_dim: int, dropout: float
    ):
        super().__init__()
        self.self_attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.cross_attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.linear1 = nn.Linear(hidden_dim, feedforward_dim)
        self.linear2 = nn.Linear(feedforward_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        self.dropout_self = nn.Dropout(dropout)
        self.dropout_cross = nn.Dropout(dropout)
        self.dropout_ffn = nn.Dropout(dropout)
        self.dropout_output = nn.Dropout(dropout)

    def forward(
        self,
        target: torch.Tensor,
        memory: torch.Tensor,
        position: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        query_key = target + position
        attended = self.self_attention(
            query_key, query_key, target, attn_mask=attention_mask, need_weights=False
        )[0]
        target = self.norm1(target + self.dropout_self(attended))

        attended = self.cross_attention(
            target + position,
            memory + position,
            memory,
            attn_mask=attention_mask,
            need_weights=False,
        )[0]
        target = self.norm2(target + self.dropout_cross(attended))

        feedforward = self.linear2(self.dropout_ffn(F.gelu(self.linear1(target))))
        return self.norm3(target + self.dropout_output(feedforward))


class ReconstructionDecoder(nn.Module):
    """Reconstruct BxCxHxW features with learned spatial queries.

    ``neighbor_size=None`` gives the unmasked baseline. Set it to an odd
    integer (7 in the full model) to restrict encoder and decoder attention
    to spatial neighbors. Feature jitter is applied only in training mode.
    """

    def __init__(
        self,
        in_channels: int = 1792,
        hidden_dim: int = 128,
        num_heads: int = 4,
        num_encoder_layers: int = 2,
        num_decoder_layers: int = 2,
        feedforward_dim: int = 512,
        dropout: float = 0.1,
        base_grid_size: Tuple[int, int] = (32, 32),
        neighbor_size: Optional[int] = None,
        jitter_scale: float = 0.0,
        jitter_probability: float = 0.0,
    ) -> None:
        super().__init__()
        if in_channels <= 0 or hidden_dim <= 0 or num_heads <= 0:
            raise ValueError("channel and attention dimensions must be positive.")
        if hidden_dim % num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads.")
        if min(base_grid_size) <= 0:
            raise ValueError("base_grid_size dimensions must be positive.")
        if num_encoder_layers < 1 or num_decoder_layers < 1:
            raise ValueError("encoder and decoder must each have at least one layer.")
        if neighbor_size is not None and (
            neighbor_size < 1 or neighbor_size % 2 == 0
        ):
            raise ValueError("neighbor_size must be a positive odd integer.")
        if jitter_scale < 0 or not 0 <= jitter_probability <= 1:
            raise ValueError("jitter_scale must be non-negative and probability in [0, 1].")

        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.base_grid_size = base_grid_size
        self.neighbor_size = neighbor_size
        self.jitter_scale = jitter_scale
        self.jitter_probability = jitter_probability
        base_h, base_w = base_grid_size

        self.input_projection = nn.Conv2d(in_channels, hidden_dim, kernel_size=1)
        self.output_projection = nn.Conv2d(hidden_dim, in_channels, kernel_size=1)
        self.position_embedding = nn.Parameter(
            torch.empty(1, hidden_dim, base_h, base_w)
        )
        self.query_embedding = nn.Parameter(
            torch.empty(1, hidden_dim, base_h, base_w)
        )
        nn.init.trunc_normal_(self.position_embedding, std=0.02)
        nn.init.trunc_normal_(self.query_embedding, std=0.02)

        self.encoder = nn.ModuleList(
            _EncoderBlock(hidden_dim, num_heads, feedforward_dim, dropout)
            for _ in range(num_encoder_layers)
        )
        self.decoder = nn.ModuleList(
            _DecoderBlock(hidden_dim, num_heads, feedforward_dim, dropout)
            for _ in range(num_decoder_layers)
        )

    def _spatial_embeddings(
        self, height: int, width: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if (height, width) == self.base_grid_size:
            return self.position_embedding, self.query_embedding
        size = (height, width)
        position = F.interpolate(
            self.position_embedding, size=size, mode="bilinear", align_corners=False
        )
        query = F.interpolate(
            self.query_embedding, size=size, mode="bilinear", align_corners=False
        )
        return position, query

    def _attention_mask(
        self, height: int, width: int, device: torch.device
    ) -> Optional[torch.Tensor]:
        if self.neighbor_size is None:
            return None
        y, x = torch.meshgrid(
            torch.arange(height, device=device),
            torch.arange(width, device=device),
            indexing="ij",
        )
        y = y.reshape(-1)
        x = x.reshape(-1)
        radius = self.neighbor_size // 2
        allowed = (
            (y[:, None] - y[None, :]).abs().le(radius)
            & (x[:, None] - x[None, :]).abs().le(radius)
        )
        return ~allowed

    def _jitter(self, features: torch.Tensor) -> torch.Tensor:
        if not self.training or self.jitter_scale == 0 or self.jitter_probability == 0:
            return features
        feature_norm = features.norm(p=2, dim=1, keepdim=True) / self.in_channels
        noise = torch.randn_like(features) * feature_norm * self.jitter_scale
        if self.jitter_probability < 1:
            apply_noise = torch.rand(
                features.shape[0], 1, 1, 1, device=features.device
            ) < self.jitter_probability
            noise = noise * apply_noise
        return features + noise

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 4 or features.shape[1] != self.in_channels:
            raise ValueError(
                f"features must have shape (batch, {self.in_channels}, height, width)."
            )
        if features.shape[0] == 0 or min(features.shape[-2:]) < 1:
            raise ValueError("feature batch and spatial dimensions must be non-empty.")

        batch, _, height, width = features.shape
        projected = self.input_projection(self._jitter(features))
        source = projected.flatten(2).transpose(1, 2)
        position_map, query_map = self._spatial_embeddings(height, width)
        position = position_map.flatten(2).transpose(1, 2)
        query = query_map.flatten(2).transpose(1, 2).expand(batch, -1, -1)
        attention_mask = self._attention_mask(height, width, features.device)

        memory = source
        for layer in self.encoder:
            memory = layer(memory, position, attention_mask)
        decoded = query
        for layer in self.decoder:
            decoded = layer(decoded, memory, position, attention_mask)

        decoded = decoded.transpose(1, 2).reshape(
            batch, self.hidden_dim, height, width
        )
        return self.output_projection(decoded)


def reconstruction_loss(
    target: torch.Tensor, reconstruction: torch.Tensor
) -> torch.Tensor:
    """Mean squared error between aligned feature maps."""
    if target.shape != reconstruction.shape:
        raise ValueError(
            "target and reconstruction must have identical shapes; "
            f"got {tuple(target.shape)} and {tuple(reconstruction.shape)}."
        )
    return F.mse_loss(reconstruction, target)
