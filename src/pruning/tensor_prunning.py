"""Utility functions for tensor (unstructured/structured) pruning."""

from __future__ import annotations

from typing import Dict, Iterable, Optional, Tuple, Type

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

__all__ = [
    "global_unstructured_prune",
    "layer_unstructured_prune",
    "structured_filter_prune",
    "random_unstructured_prune",
    "threshold_prune",
    "report_sparsity",
    "make_pruning_permanent",
]


DEFAULT_TARGET_MODULES: Tuple[Type[nn.Module], ...] = (nn.Conv2d, nn.Linear)


def _iter_prunable_modules(
    model: nn.Module,
    target_types: Iterable[Type[nn.Module]],
    exclude_name_patterns: Optional[Iterable[str]] = None,
) -> Iterable[Tuple[str, nn.Module]]:
    exclude_name_patterns = list(exclude_name_patterns or [])
    target_types = tuple(target_types)
    if not target_types:
        return

    for name, module in model.named_modules():
        if isinstance(module, target_types):
            if any(p in name for p in exclude_name_patterns):
                continue
            yield name, module


def global_unstructured_prune(
    model: nn.Module,
    amount: float = 0.3,
    *,
    target_types: Tuple[Type[nn.Module], ...] = DEFAULT_TARGET_MODULES,
    exclude_name_patterns: Optional[Iterable[str]] = ("lora_", "classifier"),
) -> None:
    """Globally prune weights with the smallest magnitude (L1 norm).

    Parameters
    ----------
    model:
        Model whose parameters will be pruned in-place.
    amount:
        Fraction of parameters to remove globally.
    target_types:
        Module types to consider (defaults to ``Conv2d`` and ``Linear``).
    exclude_name_patterns:
        Iterable with substrings; modules containing them in their ``name`` will be skipped.
    """

    if not 0.0 < amount < 1.0:
        raise ValueError("amount must be within (0, 1)")

    parameters_to_prune = [
        (module, "weight") for _, module in _iter_prunable_modules(model, target_types, exclude_name_patterns)
    ]

    if not parameters_to_prune:
        raise RuntimeError("No parameters matched the requested pruning configuration.")

    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=amount,
    )


def layer_unstructured_prune(
    model: nn.Module,
    amount: float = 0.2,
    *,
    target_types: Tuple[Type[nn.Module], ...] = DEFAULT_TARGET_MODULES,
    exclude_name_patterns: Optional[Iterable[str]] = ("lora_", "classifier"),
) -> None:
    """Apply independent L1-unstructured pruning to each eligible layer."""

    if not 0.0 < amount < 1.0:
        raise ValueError("amount must be within (0, 1)")

    for _, module in _iter_prunable_modules(model, target_types, exclude_name_patterns):
        prune.l1_unstructured(module, name="weight", amount=amount)


def structured_filter_prune(
    model: nn.Module,
    amount: float = 0.2,
    *,
    n: int = 2,
    dim: int = 0,
    target_types: Tuple[Type[nn.Module], ...] = (nn.Conv2d,),
    exclude_name_patterns: Optional[Iterable[str]] = ("lora_", "classifier"),
) -> None:
    """Structured pruning that removes complete filters/channels using ``ln_structured``.

    Parameters
    ----------
    amount:
        Fraction of channels/filters to remove from each layer.
    n:
        Norm degree used when ranking filters (``2`` corresponds to the L2 norm).
    dim:
        Dimension representing the output channels (``0`` for ``Conv2d``).
    """

    if not 0.0 < amount < 1.0:
        raise ValueError("amount must be within (0, 1)")

    for _, module in _iter_prunable_modules(model, target_types, exclude_name_patterns):
        prune.ln_structured(module, name="weight", amount=amount, n=n, dim=dim)


def random_unstructured_prune(
    model: nn.Module,
    amount: float = 0.1,
    *,
    target_types: Tuple[Type[nn.Module], ...] = DEFAULT_TARGET_MODULES,
    exclude_name_patterns: Optional[Iterable[str]] = ("lora_", "classifier"),
) -> None:
    """Randomly prune parameters in eligible layers.

    While magnitude pruning is usually preferred, random pruning is a simple
    baseline that can be used for ablation studies or as an initialization step
    before fine-tuning specialised pruning schedules.
    """

    if not 0.0 < amount < 1.0:
        raise ValueError("amount must be within (0, 1)")

    for _, module in _iter_prunable_modules(model, target_types, exclude_name_patterns):
        prune.random_unstructured(module, name="weight", amount=amount)


def threshold_prune(
    model: nn.Module,
    threshold: float,
    *,
    target_types: Tuple[Type[nn.Module], ...] = DEFAULT_TARGET_MODULES,
    exclude_name_patterns: Optional[Iterable[str]] = ("lora_", "classifier"),
    prune_bias: bool = False,
) -> None:
    """Zero-out parameters whose magnitude falls below ``threshold``.

    Parameters
    ----------
    threshold:
        Absolute value below which weights are removed.
    prune_bias:
        Apply the same thresholding to bias terms when available.
    """

    if threshold < 0:
        raise ValueError("threshold must be non-negative")

    for _, module in _iter_prunable_modules(model, target_types, exclude_name_patterns):
        weight_mask = torch.abs(module.weight) > threshold
        prune.custom_from_mask(module, name="weight", mask=weight_mask)
        if prune_bias and hasattr(module, "bias") and module.bias is not None:
            bias_mask = torch.abs(module.bias) > threshold
            prune.custom_from_mask(module, name="bias", mask=bias_mask)


def report_sparsity(model: nn.Module, target_types: Tuple[Type[nn.Module], ...] = DEFAULT_TARGET_MODULES) -> Dict[str, float]:
    """Return per-module and global sparsity statistics."""

    stats: Dict[str, float] = {}
    total_nonzero = 0
    total_elements = 0
    target_types = tuple(target_types)
    for name, module in model.named_modules():
        if isinstance(module, target_types):
            weight = module.weight.detach()
            numel = weight.numel()
            zeros = torch.count_nonzero(weight == 0).item()
            sparsity = float(zeros) / float(numel)
            stats[name] = sparsity
            total_nonzero += numel - int(zeros)
            total_elements += numel

    if total_elements:
        stats["global"] = 1.0 - (total_nonzero / float(total_elements))
    else:
        stats["global"] = 0.0
    return stats


def make_pruning_permanent(model: nn.Module) -> None:
    """Remove the pruning re-parametrisation so that zeroed weights become persistent."""

    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            if hasattr(module, "weight_mask"):
                prune.remove(module, "weight")
            if hasattr(module, "bias_mask"):
                prune.remove(module, "bias")

