"""Evidence-gated hurdle forecaster for county-month counts.

Typed feature channels are projected separately and fused with a learned
channel gate. A hurdle head predicts event occurrence and positive count
intensity. The hurdle estimate is mixed with a moving-average prior through a
monotone gate on historical support, data quality and recent stability.

Predictions are returned in log1p space, matching the training loss.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from .encoding import PositionalEncoding

CHANNEL_NAMES = ("history", "seasonality", "quality", "composition", "spatial", "other")


def build_feature_channel_indices(feature_columns: Sequence[str]) -> dict[str, list[int]]:
    """Assign each non-UST feature column to one typed channel."""

    groups: dict[str, list[int]] = {name: [] for name in CHANNEL_NAMES}
    for index, name in enumerate(feature_columns):
        lowered = str(name).lower()
        if lowered.startswith("ust_"):
            continue
        if lowered.startswith("type_"):
            channel = "composition"
        elif "valid_rate" in lowered or "parse_rate" in lowered or lowered.endswith("_quality"):
            channel = "quality"
        elif lowered.startswith("month_") or lowered.startswith("bucket_ratio_") or lowered in {"month", "year"}:
            channel = "seasonality"
        elif lowered.startswith(("fine_level_", "space_", "geo_", "spatial_")):
            channel = "spatial"
        elif lowered.startswith(("lag_", "rolling_", "growth_", "bucket_count_")) or "case_count" in lowered:
            channel = "history"
        else:
            channel = "other"
        groups[channel].append(index)
    return groups


class EvidenceGatedHurdleForecaster(nn.Module):
    """Multi-channel Transformer with a monotone evidence gate.

    ``gate_features`` holds historical support, data quality and recent
    stability, each in ``[0, 1]``. Slopes are parameterized with softplus, so
    raising one evidence value cannot lower the weight of the hurdle branch.
    """

    def __init__(
        self,
        feature_dim: int,
        ust_dim: int,
        horizons: int,
        num_types: int,
        channel_indices: Mapping[str, Sequence[int]] | None = None,
        model_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        gate_feature_dim: int = 3,
        gate_floor: float = 0.0,
        gate_ceiling: float = 1.0,
        event_positive_weight: float = 1.0,
        gate_regularization_weight: float = 0.05,
    ) -> None:
        super().__init__()
        if gate_feature_dim != 3:
            raise ValueError("the evidence gate expects support, quality and stability")
        if not 0.0 <= gate_floor < gate_ceiling <= 1.0:
            raise ValueError("require 0 <= gate_floor < gate_ceiling <= 1")
        if event_positive_weight < 1.0:
            raise ValueError("event_positive_weight must be at least one")
        if gate_regularization_weight < 0.0:
            raise ValueError("gate_regularization_weight must be non-negative")
        self.gate_floor = float(gate_floor)
        self.gate_ceiling = float(gate_ceiling)
        self.register_buffer("event_positive_weight", torch.tensor(float(event_positive_weight)))
        self.register_buffer("gate_regularization_weight", torch.tensor(float(gate_regularization_weight)))
        self.ust_dim = int(ust_dim)
        self.horizons = int(horizons)
        self.num_types = int(num_types)

        provided = channel_indices or build_feature_channel_indices([f"feature_{i}" for i in range(feature_dim)])
        self.channel_indices = {name: [int(index) for index in provided.get(name, [])] for name in CHANNEL_NAMES}
        if any(index < 0 or index >= feature_dim for indices in self.channel_indices.values() for index in indices):
            raise ValueError("channel_indices contains an index outside feature_dim")

        self.channel_projections = nn.ModuleDict()
        for name in CHANNEL_NAMES:
            width = len(self.channel_indices[name]) + (self.ust_dim if name == "spatial" else 0)
            self.channel_projections[name] = nn.Linear(max(1, width), model_dim)
        self.channel_fusion = nn.Sequential(
            nn.Linear(len(CHANNEL_NAMES) * model_dim, model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim, len(CHANNEL_NAMES)),
        )
        self.full_state_projection = nn.Linear(feature_dim + self.ust_dim, model_dim)
        self.position = PositionalEncoding(model_dim=model_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=model_dim * 2,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(model_dim)

        self.event_head = nn.Linear(model_dim, self.horizons)
        self.positive_head = nn.Linear(model_dim, self.horizons)
        self.type_head = nn.Linear(model_dim, self.num_types)

        self.gate_bias = nn.Parameter(torch.tensor(0.0))
        self.gate_raw_weights = nn.Parameter(torch.full((gate_feature_dim,), 0.5))

    def _channel_input(self, name: str, features: torch.Tensor, ust_features: torch.Tensor | None) -> torch.Tensor:
        pieces = []
        if self.channel_indices[name]:
            pieces.append(features[..., self.channel_indices[name]])
        if name == "spatial" and self.ust_dim > 0 and ust_features is not None:
            pieces.append(ust_features)
        if not pieces:
            return features.new_zeros((*features.shape[:2], 1))
        return torch.cat(pieces, dim=-1)

    def forward(self, batch: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        features = batch["features"].float()
        ust_features = batch.get("ust_features")
        if ust_features is not None:
            ust_features = ust_features.float()

        channel_representations = [
            self.channel_projections[name](self._channel_input(name, features, ust_features)) for name in CHANNEL_NAMES
        ]
        channel_stack = torch.stack(channel_representations, dim=2)
        channel_logits = self.channel_fusion(channel_stack.reshape(features.size(0), features.size(1), -1))
        channel_weights = torch.softmax(channel_logits, dim=-1)
        hidden = (channel_stack * channel_weights.unsqueeze(-1)).sum(dim=2)

        full_state = features
        if self.ust_dim > 0 and ust_features is not None:
            full_state = torch.cat([full_state, ust_features], dim=-1)
        hidden = hidden + 0.5 * self.full_state_projection(full_state)
        pooled = self.norm(self.encoder(self.position(hidden))[:, -1, :])

        event_logit = self.event_head(pooled)
        event_prob = torch.sigmoid(event_logit)
        positive_log_count = F.softplus(self.positive_head(pooled))
        structured_count = event_prob * torch.expm1(positive_log_count).clamp(min=0.0, max=1e6)

        prior = batch.get("count_prior")
        if prior is None:
            prior = features.new_zeros((features.size(0), self.horizons))
        prior = prior.float().clamp(min=0.0)

        evidence = batch.get("gate_features")
        if evidence is None:
            evidence = features.new_full((features.size(0), 3), 0.5)
        evidence = evidence.float().clamp(0.0, 1.0)
        gate_weights = F.softplus(self.gate_raw_weights)
        gate = torch.sigmoid(self.gate_bias + ((evidence - 0.5) * gate_weights).sum(dim=-1))
        gate = self.gate_floor + (self.gate_ceiling - self.gate_floor) * gate
        final_count = (1.0 - gate.unsqueeze(-1)) * prior + gate.unsqueeze(-1) * structured_count

        return {
            "count_pred": torch.log1p(final_count.clamp(min=0.0)),
            "type_pred": self.type_head(pooled),
            "event_logit": event_logit,
            "event_prob": event_prob,
            "event_positive_weight": self.event_positive_weight,
            "gate_regularization_weight": self.gate_regularization_weight,
            "positive_log_count": positive_log_count,
            "structured_count": structured_count,
            "gate": gate,
            "gate_weights": gate_weights,
            "channel_weights": channel_weights[:, -1, :],
        }
