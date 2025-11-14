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
    "progressive_channel_pruning",
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
        registry = {
            "l1": lambda: tp.importance.MagnitudeImportance(p=1),
            "l2": lambda: tp.importance.MagnitudeImportance(p=2),
            "taylor": tp.importance.TaylorImportance,
            "fpgm": tp.importance.FPGMImportance,
        }
        if norm not in registry:
            available = "', '".join(sorted(registry))
            raise ValueError(
                f"importance must be one of '{available}' or a torch_pruning.Importance instance"
            )
        return registry[norm]()
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
    # for pat in (r"\.base$", r"lora_", r"^model\.classifier"):
    #     if pat not in exclude_name_patterns:
    #         exclude_name_patterns.append(pat)

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


def progressive_channel_pruning(
    model: nn.Module,
    example_inputs: TensorOrArgs,
    schedule: Sequence[float],
    *,
    warmup_steps: int = 0,
    **kwargs: Any,
) -> nn.Module:
    """Iteratively prune a model following an increasing sparsity ``schedule``.

    This helper repeatedly applies :func:`prune_model_channels`, which is often
    recommended over a single aggressive pruning step for improved stability.

    Parameters
    ----------
    schedule:
        Iterable with target sparsity values (between ``0`` and ``1``). Values
        should be sorted in ascending order. Each step reuses the pruned model
        from the previous iteration.
    warmup_steps:
        Number of initial schedule entries that will be skipped. This can be
        useful when resuming progressive pruning and the early steps have
        already been applied.
    kwargs:
        Additional keyword arguments forwarded to :func:`prune_model_channels`.
    """

    values = list(schedule)
    if not values:
        raise ValueError("schedule must contain at least one sparsity value")

    if warmup_steps < 0:
        raise ValueError("warmup_steps must be >= 0")

    start_idx = min(len(values), warmup_steps)
    if start_idx >= len(values):
        return model

    last_idx = len(values) - 1
    for idx, amount in enumerate(values[start_idx:], start=start_idx):
        if not 0.0 < amount < 1.0:
            raise ValueError("All sparsity values in schedule must be within (0, 1)")
        verbose = kwargs.get("verbose", True) and idx == last_idx
        prune_model_channels(
            model,
            example_inputs,
            amount=amount,
            verbose=verbose,
            **{k: v for k, v in kwargs.items() if k != "verbose"},
        )
    return model



import re
import torch
import torch.nn as nn
import torch.ao.nn.quantized as nnq
import torch_pruning as tp

# ---------- 1) helpers ----------
def _is_depthwise(m: nn.Conv2d) -> bool:
    return isinstance(m, nn.Conv2d) and m.groups == m.in_channels == m.out_channels

def _collect_ignored_layers(model: nn.Module, exclude_name_patterns):
    pats = [re.compile(p) for p in exclude_name_patterns]
    ignored = set()
    for name, m in model.named_modules():
        if any(p.search(name) for p in pats):
            ignored.add(m)
    # глубинные conv лучше не трогать
    for m in model.modules():
        if _is_depthwise(m):
            ignored.add(m)
    return list(ignored)

def _conv_summary(model):
    rows = []
    for n, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            rows.append((n, m.in_channels, m.out_channels, m.groups))
    return rows

# ---------- 2) ref <-> backend конвертеры c fallback по версиям ----------
def to_reference_from_backend(qmodel: nn.Module) -> nn.Module:
    """
    Превращает backend-quantized граф (с quantized ops) в reference-квант:
    float Conv/Linear + стобы Quant/DeQuant. Сохраняет текущие scale/zero_point.
    """
    # В разных версиях API называется по-разному. Пробуем варианты.
    # PyTorch 2.1+:
    try:
        from torch.ao.quantization.fx._lower_to_native_backend import convert_to_reference_decomposed
        return convert_to_reference_decomposed(qmodel)
    except Exception:
        pass
    # PyTorch 2.0 / 1.13 (FX):
    try:
        from torch.ao.quantization.quantize_fx import convert_to_reference_fx
        return convert_to_reference_fx(qmodel)
    except Exception as e:
        raise RuntimeError(
            "Не удалось конвертировать модель в reference-режим. "
            "Нужна версия PyTorch с FX reference quant APIs."
        ) from e

def lower_reference_to_backend(ref_model: nn.Module) -> nn.Module:
    """
    Опускает reference-квант обратно в backend-квант (qnnpack/fbgemm).
    """
    # PyTorch 2.1+: lower_to_native_backend
    try:
        from torch.ao.quantization.fx._lower_to_native_backend import lower_to_native_backend
        return lower_to_native_backend(ref_model)
    except Exception:
        pass
    # PyTorch 2.0 / 1.13: convert_fx(..., is_reference=True)
    try:
        from torch.ao.quantization.quantize_fx import convert_fx
        return convert_fx(ref_model)
    except Exception as e:
        raise RuntimeError(
            "Не удалось опустить reference-модель в backend-квант. "
            "Уточни версию PyTorch: подскажу точный вызов."
        ) from e

# ---------- 3) основной пайплайн ----------
def prune_quantized_model(qmodel: nn.Module,
                          example_inputs,
                          amount: float = 0.30,
                          min_out_channels: int = 8,
                          verbose: bool = True) -> nn.Module:
    # 1) если модель backend-quantized — переводим в reference (float Conv/Linear + Quant/DeQuant)
    if any(isinstance(m, (nnq.Conv2d, nnq.Linear)) for m in qmodel.modules()):
        if verbose:
            print("[info] Детектирован backend-quantized граф → переводим в reference…")
        ref = to_reference_from_backend(qmodel)   # твоя функция; должна уметь это делать на твоей версии PT
    else:
        ref = qmodel

    ref.eval()

    # 2) диагностика до
    before = _conv_summary(ref)
    if verbose:
        print("[ref] Conv2d BEFORE:")
        for n, ci, co, g in before[:12]:
            print(f"  {n:40s} in={ci:4d} out={co:4d} groups={g}")
        if len(before) > 12:
            print(f"  ... total convs: {len(before)}")

    # 3) исключения: LoRA/adapter/Dropout/BatchNorm и строгие depthwise
    exclude = [  r"\.dropout", r"\.bn", r"BatchNorm"]
    ignored_layers = _collect_ignored_layers(ref, exclude)

    importance = tp.importance.MagnitudeImportance(p=1)

    # 4) формируем per-layer sparsity с учётом min_out_channels (и для Linear — по out_features)
    ch_sparsity_dict = {}
    for m in ref.modules():
        if isinstance(m, nn.Conv2d) and not _is_depthwise(m):
            max_spr = max(0.0, 1.0 - (min_out_channels / float(m.out_channels)))
            tgt = min(amount, max_spr)
            if tgt > 0.0:
                ch_sparsity_dict[m] = float(tgt)
        elif isinstance(m, nn.Linear):
            max_spr = max(0.0, 1.0 - (min_out_channels / float(m.out_features)))
            tgt = min(amount, max_spr)
            if tgt > 0.0:
                ch_sparsity_dict[m] = float(tgt)

    # 5) создаём pruner БЕЗ global_pruning, но с ch_sparsity_dict
    pruner = tp.pruner.MagnitudePruner(
        ref,
        example_inputs=example_inputs,         # тензор или tuple — как в твоём форварде
        importance=importance,
        global_pruning=False,                  # критично: False
        ch_sparsity_dict=ch_sparsity_dict,     # наши цельовые sparsity по слоям
        ignored_layers=ignored_layers,
        root_module_types=[nn.Conv2d, nn.Linear],
    )

    pruner.step()

    # 6) диагностика после
    after = _conv_summary(ref)
    if verbose:
        print("[ref] Conv2d AFTER:")
        for (n1, ci1, co1, g1), (_, _, co2, _) in zip(before, after):
            mark = " <-- pruned" if co2 != co1 else ""
            print(f"  {n1:40s} out {co1:4d} -> {co2:4d}{mark}")
        total_before = sum(co for _, _, co, _ in before)
        total_after  = sum(co for _, _, co, _ in after)
        print(f"[ref] total out_channels: {total_before} -> {total_after} (Δ={total_before-total_after})")

    # 7) возвращаемся в backend-квант
    q_pruned = lower_reference_to_backend(ref).eval()
    return q_pruned

