# Optimization/src/pruning/channel_pruning.py  (v2)
from __future__ import annotations

import contextlib
import re
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch_pruning as tp  # v2.x

__all__ = [
    "collect_ignored_layers",
    "prune_model_channels",
]


TensorOrArgs = Union[torch.Tensor, Sequence[Any], Mapping[str, Any]]


def is_depthwise(conv: nn.Conv2d) -> bool:
    """Return ``True`` if the convolution is depthwise."""

    return isinstance(conv, nn.Conv2d) and conv.groups == conv.in_channels == conv.out_channels


def is_pointwise(conv: nn.Conv2d) -> bool:
    """Return ``True`` if the convolution is a standard 1×1 convolution."""

    return isinstance(conv, nn.Conv2d) and conv.kernel_size == (1, 1) and conv.groups == 1


def name_is_excluded(name: str, patterns: Iterable[str]) -> bool:
    """Check whether ``name`` matches any of the regular expression ``patterns``."""

    return any(re.search(p, name) for p in patterns) if patterns else False


def collect_ignored_layers(model: nn.Module, exclude_name_patterns: List[str]) -> List[nn.Module]:
    """Collect layers that should be ignored by the structured channel pruner."""

    ignored: List[nn.Module] = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) and is_depthwise(module):
            ignored.append(module)
            continue
        if name_is_excluded(name, exclude_name_patterns):
            ignored.append(module)
    return ignored


def _normalize_example_inputs(example_inputs: TensorOrArgs, device: torch.device | str):
    """Move provided example inputs to the target device if they are tensors."""

    if isinstance(example_inputs, torch.Tensor):
        return example_inputs.to(device)
    if isinstance(example_inputs, Mapping):
        return {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in example_inputs.items()}
    if isinstance(example_inputs, (list, tuple)):
        processed = []
        for item in example_inputs:
            if isinstance(item, torch.Tensor):
                processed.append(item.to(device))
            else:
                processed.append(item)
        return tuple(processed)
    raise TypeError(
        "example_inputs must be a Tensor, sequence of tensors or mapping compatible with model forward"
    )


def _resolve_importance(importance: Union[str, tp.importance.Importance]) -> tp.importance.Importance:
    if isinstance(importance, str):
        norm = importance.lower()
        if norm not in {"l1", "l2"}:
            raise ValueError("importance must be either 'l1', 'l2' or a torch_pruning.Importance instance")
        p = 1 if norm == "l1" else 2
        return tp.importance.MagnitudeImportance(p=p)
    if isinstance(importance, tp.importance.Importance):
        return importance
    raise TypeError("importance must be either a string or torch_pruning.importance.Importance instance")


def prune_model_channels(
    model: nn.Module,
    example_inputs: TensorOrArgs,
    amount: float = 0.30,
    *,
    exclude_name_patterns: Optional[List[str]] = None,
    min_out_channels: int = 8,
    importance: Union[str, tp.importance.Importance] = "l1",
    device: str = "cpu",
    global_pruning: bool = False,
    verbose: bool = True,
) -> nn.Module:
    """Structured channel pruning using *torch-pruning* 2.x.

    Parameters
    ----------
    model:
        Model to prune **in-place**. A copy can be supplied by calling ``deepcopy`` upstream.
    example_inputs:
        Sample inputs required by torch-pruning to trace the computation graph.
    amount:
        Desired sparsity per-layer (default 30%).
    exclude_name_patterns:
        Regex patterns for modules that should be untouched (LoRA, heads, etc.).
    min_out_channels:
        Minimal number of output channels to retain for each convolution.
    importance:
        Either ``"l1"``, ``"l2"`` or a custom ``torch_pruning`` importance object.
    device:
        Device used when tracing the model with the example inputs.
    global_pruning:
        Whether to prune channels globally instead of per-layer. When enabled ``amount``
        represents global sparsity.
    verbose:
        Print a short summary with the number of pruned modules.
    """

    if not 0.0 < amount < 1.0:
        raise ValueError("amount must be within (0, 1)")

    min_out_channels = max(1, int(min_out_channels))

    model.eval().to(device)
    normalized_inputs = _normalize_example_inputs(example_inputs, device)

    importance_fn = _resolve_importance(importance)

    exclude_name_patterns = list(exclude_name_patterns or [])
    for pat in (r"\.base$", r"lora_", r"^model\.classifier"):
        if pat not in exclude_name_patterns:
            exclude_name_patterns.append(pat)

    ignored_layers = collect_ignored_layers(model, exclude_name_patterns)

    prunable: List[nn.Conv2d] = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) and module.groups == 1 and not is_depthwise(module):
            if not name_is_excluded(name, exclude_name_patterns):
                prunable.append(module)

    if not prunable:
        if verbose:
            print("[channel_pruner] No eligible Conv2d modules to prune.")
        return model.to("cpu").eval()

    ch_sparsity_dict: Dict[nn.Module, float] = {}
    if not global_pruning:
        for module in prunable:
            ch = module.out_channels
            max_sparsity = max(0.0, 1.0 - (min_out_channels / float(ch)))
            target = min(amount, max_sparsity)
            if target > 0.0:
                ch_sparsity_dict[module] = float(target)

        if not ch_sparsity_dict:
            if verbose:
                print("[channel_pruner] Nothing to prune given min_out_channels constraints.")
            return model.to("cpu").eval()

    # torch-pruning builds computation graph using gradients - ensure grad-mode enabled temporarily
    grad_ctx = contextlib.nullcontext()
    if not torch.is_grad_enabled():
        grad_ctx = torch.enable_grad()

    with grad_ctx:
        pruner = tp.pruner.MagnitudePruner(
            model,
            example_inputs=normalized_inputs,
            importance=importance_fn,
            global_pruning=global_pruning,
            ch_sparsity=amount if global_pruning else None,
            ch_sparsity_dict=None if global_pruning else ch_sparsity_dict,
            ignored_layers=ignored_layers,
            root_module_types=[nn.Conv2d],
        )
        pruner.step()

    model.to("cpu").eval()

    if verbose:
        pruned = len(prunable) if global_pruning else len(ch_sparsity_dict)
        print(f"[channel_pruner] Pruned {pruned} convolutional modules (global={global_pruning}).")

    return model
