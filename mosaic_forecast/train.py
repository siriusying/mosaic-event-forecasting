"""Training, prediction and the validation-only endpoint safeguard."""

from __future__ import annotations

import copy
import json
import random
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .features import multiscale_prior, prepare_arrays, resolve_feature_columns, standardize
from .metrics import rmse, row_mean
from .model import EvidenceGatedHurdleForecaster, build_feature_channel_indices


@dataclass(frozen=True)
class TrainConfig:
    seed: int = 37
    epochs: int = 25
    learning_rate: float = 1e-3
    feature_mode: str = "leakage_safe"
    model_dim: int = 64
    num_heads: int = 4
    num_layers: int = 2
    dropout: float = 0.1
    gate_floor: float = 0.0
    gate_ceiling: float = 1.0
    device: str = "cpu"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def _normalised_types(values: torch.Tensor) -> torch.Tensor:
    clipped = values.float().clamp(min=1e-8)
    return clipped / clipped.sum(dim=-1, keepdim=True).clamp(min=1e-8)


def training_loss(output: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> torch.Tensor:
    """Count, hurdle and type losses used for every reported model."""

    target_log = torch.log1p(batch["count_targets"].float().clamp(min=0.0))
    type_target = _normalised_types(batch["type_targets"])
    type_loss = F.kl_div(torch.log_softmax(output["type_pred"], dim=-1), type_target, reduction="batchmean")

    positive_target = (batch["count_targets"] > 0.0).float()
    event_loss = F.binary_cross_entropy_with_logits(
        output["event_logit"],
        positive_target,
        pos_weight=output["event_positive_weight"].to(positive_target.device),
    )
    mask = positive_target.bool()
    if mask.any():
        positive_loss = F.smooth_l1_loss(output["positive_log_count"][mask], target_log[mask])
    else:
        positive_loss = output["positive_log_count"].sum() * 0.0
    count_loss = F.smooth_l1_loss(output["count_pred"], target_log)

    gate = output["gate"].float().reshape(-1)
    gate_mean = gate.mean()
    gate_std = gate.std(unbiased=False) if gate.numel() > 1 else gate.new_zeros(())
    gate_penalty = (gate_mean - 0.5).pow(2) + F.relu(0.05 - gate_std).pow(2)

    return (
        count_loss
        + 0.5 * event_loss
        + 0.5 * positive_loss
        + 0.2 * type_loss
        + output["gate_regularization_weight"] * gate_penalty
    )


def fit(model: nn.Module, train_batch: dict[str, torch.Tensor], val_batch: dict[str, torch.Tensor], epochs: int, lr: float) -> dict[str, Any]:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_state = copy.deepcopy(model.state_dict())
    best_val = float("inf")
    history = []
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        loss = training_loss(model(train_batch), train_batch)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = float(training_loss(model(val_batch), val_batch).item())
        history.append({"epoch": epoch + 1, "train_loss": float(loss.detach().item()), "val_loss": val_loss})
        if val_loss < best_val:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return {"best_val_loss": best_val, "train_size": int(train_batch["features"].shape[0]), "loss_history": history}


@torch.no_grad()
def predict(model: nn.Module, batch: dict[str, torch.Tensor]) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    output = model(batch)
    counts = torch.expm1(output["count_pred"]).clamp(min=0.0).cpu().numpy()
    types = torch.softmax(output["type_pred"], dim=-1).cpu().numpy()
    return counts, types


@torch.no_grad()
def gate_variants(model: nn.Module, batch: dict[str, torch.Tensor]) -> dict[str, np.ndarray]:
    model.eval()
    output = model(batch)
    structured = output["structured_count"].float()
    prior = batch["count_prior"].float().clamp(min=0.0)
    gate = output["gate"].float().unsqueeze(-1)
    full = (1.0 - gate) * prior + gate * structured
    return {
        "full": full.cpu().numpy(),
        "structured": structured.cpu().numpy(),
        "prior": prior.cpu().numpy(),
        "gate": output["gate"].cpu().numpy(),
    }


def select_endpoint(val: dict[str, np.ndarray], test: dict[str, np.ndarray], val_target: np.ndarray) -> tuple[str, np.ndarray, np.ndarray, dict[str, float]]:
    """Choose the full gate or the hurdle endpoint on validation rows."""

    scores = {
        "full_gate": row_mean(rmse, val_target, val["full"]),
        "hurdle_endpoint": row_mean(rmse, val_target, val["structured"]),
    }
    selected = min(scores, key=scores.get)
    key = "full" if selected == "full_gate" else "structured"
    return selected, val[key], test[key], {name: float(value) for name, value in scores.items()}


def _batch(arrays: dict[str, np.ndarray], mask: np.ndarray, device: torch.device) -> dict[str, torch.Tensor]:
    def tensor(name: str) -> torch.Tensor:
        return torch.tensor(arrays[name][mask], dtype=torch.float32, device=device)

    return {
        "features": tensor("features"),
        "ust_features": tensor("ust_features"),
        "count_targets": tensor("targets"),
        "count_prior": tensor("prior"),
        "type_targets": tensor("types"),
        "history_last": tensor("history_last"),
        "gate_features": tensor("gate"),
    }


def build_model(arrays: dict[str, np.ndarray], config: TrainConfig) -> EvidenceGatedHurdleForecaster:
    feature_names = [str(name) for name in arrays["feature_columns"] if not str(name).startswith("ust_")]
    return EvidenceGatedHurdleForecaster(
        feature_dim=int(arrays["features"].shape[-1]),
        ust_dim=int(arrays["ust_features"].shape[-1]),
        horizons=int(arrays["targets"].shape[-1]),
        num_types=int(arrays["types"].shape[-1]),
        channel_indices=build_feature_channel_indices(feature_names),
        model_dim=config.model_dim,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        dropout=config.dropout,
        gate_floor=config.gate_floor,
        gate_ceiling=config.gate_ceiling,
    )


def prepare_training_data(samples, config: TrainConfig) -> tuple[dict[str, np.ndarray], list[str]]:
    raw_features = list(json.loads(samples.iloc[0]["feature_columns"]))
    feature_columns = resolve_feature_columns(raw_features, config.feature_mode)
    arrays = prepare_arrays(samples, feature_columns)
    return arrays, feature_columns


def run_forward(samples, split: dict[str, list[str]], config: TrainConfig) -> dict[str, Any]:
    """Train one model on the forward split and evaluate the test months."""

    device = resolve_device(config.device)
    set_seed(config.seed)
    arrays, feature_columns = prepare_training_data(samples, config)
    labels = samples["forecast_origin"].astype(str).to_numpy()
    train_mask = np.isin(labels, split["train"])
    val_mask = np.isin(labels, split["val"])
    test_mask = np.isin(labels, split["test"])
    if not train_mask.any() or not val_mask.any() or not test_mask.any():
        raise ValueError("forward split produced an empty train, validation or test set")

    prior, prior_name = multiscale_prior(arrays["counts"], arrays["targets"], val_mask)
    arrays["prior"] = prior
    arrays["features"] = standardize(train_mask, arrays["features"])
    arrays["ust_features"] = standardize(train_mask, arrays["ust_features"])

    train_batch = _batch(arrays, train_mask, device)
    val_batch = _batch(arrays, val_mask, device)
    test_batch = _batch(arrays, test_mask, device)

    model = build_model(arrays, config).to(device)
    info = fit(model, train_batch, val_batch, config.epochs, config.learning_rate)
    val_predictions, _ = predict(model, val_batch)
    test_predictions, test_types = predict(model, test_batch)

    val_variants = gate_variants(model, val_batch)
    test_variants = gate_variants(model, test_batch)
    endpoint, val_final, test_final, endpoint_scores = select_endpoint(val_variants, test_variants, arrays["targets"][val_mask])

    return {
        "config": config.__dict__,
        "feature_columns": feature_columns,
        "prior_selected_on_validation": prior_name,
        "endpoint_selected_on_validation": endpoint,
        "endpoint_validation_scores": endpoint_scores,
        "split_sizes": {"train": int(train_mask.sum()), "val": int(val_mask.sum()), "test": int(test_mask.sum())},
        "training": info,
        "gate_test": {
            "mean": float(np.mean(test_variants["gate"])),
            "min": float(np.min(test_variants["gate"])),
            "max": float(np.max(test_variants["gate"])),
        },
        "val_predictions": val_final,
        "test_predictions": test_final,
        "test_types": test_types,
        "val_full_gate": val_predictions,
        "test_mask": test_mask,
        "val_mask": val_mask,
        "arrays": arrays,
        "model": model,
    }
