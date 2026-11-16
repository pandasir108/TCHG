from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F


def operator_diversity_loss(pred: Dict[str, torch.Tensor]) -> torch.Tensor:

    val = pred.get("operator_diversity")
    if val is None:
        return torch.zeros((), device=pred["trust_logit"].device)
    return val


def balanced_bce_with_logits(
    logit: torch.Tensor,
    label: torch.Tensor,
    enabled: bool = True,
    focal_gamma: float = 0.0,
    class_weight_power: float = 1.0,
) -> torch.Tensor:

    loss = F.binary_cross_entropy_with_logits(logit, label, reduction="none")
    if enabled:
        pos = label.sum().clamp_min(1.0)
        neg = (label.numel() - label.sum()).clamp_min(1.0)
        total = label.new_tensor(float(label.numel()))
        pos_w = (total / (2.0 * pos)).pow(class_weight_power)
        neg_w = (total / (2.0 * neg)).pow(class_weight_power)
        weight = torch.where(label > 0.5, pos_w, neg_w)
        loss = loss * weight
    if focal_gamma > 0.0:
        prob = torch.sigmoid(logit)
        pt = torch.where(label > 0.5, prob, 1.0 - prob).clamp(1e-6, 1.0)
        loss = loss * (1.0 - pt).pow(focal_gamma)
    return loss.mean()


def pairwise_ranking_loss(
    logit: torch.Tensor,
    label: torch.Tensor,
    margin: float = 0.0,
    max_pairs: int = 200_000,
    negative_mask: torch.Tensor | None = None,
) -> torch.Tensor:

    pos = logit[label > 0.5]
    neg_event_mask = label <= 0.5
    if negative_mask is not None:
        hard_mask = negative_mask.bool() & neg_event_mask
        neg = logit[hard_mask] if hard_mask.any() else logit[neg_event_mask]
    else:
        neg = logit[neg_event_mask]
    if pos.numel() == 0 or neg.numel() == 0:
        return torch.zeros((), device=logit.device, dtype=logit.dtype)
    pair_count = pos.numel() * neg.numel()
    if pair_count > max_pairs:
        gen = torch.Generator(device=logit.device)
        gen.manual_seed(0)
        pos = pos[torch.randint(pos.numel(), (max_pairs,), generator=gen, device=logit.device)]
        neg = neg[torch.randint(neg.numel(), (max_pairs,), generator=gen, device=logit.device)]
        diff = pos - neg
    else:
        diff = pos[:, None] - neg[None, :]
    return F.softplus(margin - diff).mean()


def trust_calibration_loss(
    pred: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    train_cfg: Dict,
    model_cfg: Dict | None = None,
) -> tuple[torch.Tensor, Dict[str, float]]:

    label = batch["label"].float()
    trust_logit = pred["trust_logit"]
    trust_pred = pred["trust_pred"].clamp(1e-6, 1.0 - 1e-6)
    uncertainty_score = pred["uncertainty_pred"]
    memory_u = pred.get("memory_uncertainty", torch.zeros(label.shape[0], 3, device=label.device, dtype=label.dtype)).mean(dim=-1)

    trust_loss = balanced_bce_with_logits(
        trust_logit,
        label,
        enabled=bool(train_cfg.get("class_balance", True)),
        focal_gamma=float(train_cfg.get("focal_gamma", 0.0)),
        class_weight_power=float(train_cfg.get("class_weight_power", 1.0)),
    )
    lambda_rank = float(train_cfg.get("lambda_rank", 0.0))
    rank = (
        pairwise_ranking_loss(
            trust_logit,
            label,
            margin=float(train_cfg.get("rank_margin", 0.0)),
            max_pairs=int(train_cfg.get("rank_max_pairs", 200_000)),
            negative_mask=batch.get("rank_negative_mask"),
        )
        if lambda_rank > 0.0
        else torch.zeros((), device=trust_logit.device, dtype=trust_logit.dtype)
    )
    brier = torch.mean((trust_pred - label) ** 2)
    confidence = 2.0 * torch.abs(trust_pred - 0.5)
    unc_reg = torch.relu(confidence + memory_u - 1.0).mean()
    err = torch.abs(trust_pred.detach() - label)
    uerr = F.mse_loss(uncertainty_score, err)
    diversity = operator_diversity_loss(pred)
    use_calibration = True if model_cfg is None else bool(model_cfg.get("use_uncertainty_calibration", True))
    lambda_brier = float(train_cfg.get("lambda_brier", 0.2)) if use_calibration else 0.0
    lambda_unc = float(train_cfg.get("lambda_unc", 0.2)) if use_calibration else 0.0
    lambda_uerr = float(train_cfg.get("lambda_uerr", 0.1)) if use_calibration else 0.0
    total = (
        trust_loss
        + lambda_brier * brier
        + lambda_unc * unc_reg
        + lambda_uerr * uerr
        + lambda_rank * rank
        + float(train_cfg.get("lambda_div", 0.0)) * diversity
    )
    parts = {
        "loss": float(total.detach().cpu()),
        "trust_loss": float(trust_loss.detach().cpu()),
        "ranking_loss": float(rank.detach().cpu()),
        "brier_loss": float(brier.detach().cpu()),
        "uncertainty_reg": float(unc_reg.detach().cpu()),
        "uncertainty_error_loss": float(uerr.detach().cpu()),
        "operator_diversity_loss": float(diversity.detach().cpu()),
    }
    return total, parts
