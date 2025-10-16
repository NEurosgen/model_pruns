# Optimization/src/quant/weight_quant.py
from __future__ import annotations

import copy
import re
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch.ao.quantization import (
    QConfig,
    QConfigMapping,
    convert_fx,
    get_default_qconfig,
    prepare_fx,
)
from torch.ao.quantization.observer import MinMaxObserver, PerChannelMinMaxObserver

__all__ = [
    "name_is_excluded",
    "measure_latency",
    "dynamic_quantize_linear",
    "ptq_static_int8",
    "qat_int8_skeleton",
    "fuse_model",
    "quantize_fp16",
    "quantize_bf16",
    "apply_smooth_quant",
]


# -------- utils --------
def name_is_excluded(name: str, patterns: Iterable[str]) -> bool:
    """Return ``True`` if *name* matches any of the supplied regex patterns."""

    return any(re.search(p, name) for p in (patterns or []))


@torch.no_grad()
def measure_latency(
    model: nn.Module,
    input_size: Tuple[int, int, int, int] = (1, 3, 224, 224),
    iters: int = 30,
    warmup: int = 10,
    device: str = "cpu",
) -> float:
    """Measure average latency (seconds) for ``iters`` runs with a warmup phase."""

    import time

    model.eval().to(device)
    x = torch.randn(*input_size, device=device)
    for _ in range(warmup):
        _ = model(x)
    t0 = time.time()
    for _ in range(iters):
        _ = model(x)
    t1 = time.time()
    return (t1 - t0) / max(1, iters)


def _collect_fuse_candidates(module: nn.Module, prefix: str = "") -> List[List[str]]:
    """Heuristically collect ``Conv2d+BatchNorm(+ReLU)`` patterns for fusion."""

    fuse_list: List[List[str]] = []
    for name, child in module.named_children():
        child_prefix = f"{prefix}{name}"
        if isinstance(child, nn.Sequential):
            sub_names = list(child._modules.keys())
            idx = 0
            while idx < len(sub_names) - 1:
                first_name = sub_names[idx]
                second_name = sub_names[idx + 1]
                first = child._modules[first_name]
                second = child._modules[second_name]
                triple_name = None
                if isinstance(first, nn.Conv2d) and isinstance(second, nn.BatchNorm2d):
                    if idx + 2 < len(sub_names):
                        third_name = sub_names[idx + 2]
                        third = child._modules[third_name]
                        if isinstance(third, (nn.ReLU, nn.ReLU6)):
                            triple_name = third_name
                    if triple_name:
                        fuse_list.append([
                            f"{child_prefix}.{first_name}",
                            f"{child_prefix}.{second_name}",
                            f"{child_prefix}.{triple_name}",
                        ])
                        idx += 3
                        continue
                    fuse_list.append([
                        f"{child_prefix}.{first_name}",
                        f"{child_prefix}.{second_name}",
                    ])
                    idx += 2
                    continue
                idx += 1
        fuse_list.extend(_collect_fuse_candidates(child, prefix=f"{child_prefix}."))
    return fuse_list


def fuse_model(
    model: nn.Module,
    fuse_map: Optional[Sequence[Sequence[str]]] = None,
    *,
    inplace: bool = True,
) -> nn.Module:
    """Fuse adjacent ``Conv-BN-(ReLU)`` blocks before quantization.

    Parameters
    ----------
    fuse_map:
        Optional additional list of module name groups to fuse. When ``None`` the function
        performs a best-effort heuristic search for common fusion patterns.
    inplace:
        If ``False`` a cloned model is returned, leaving the original untouched.
    """

    target = model if inplace else copy.deepcopy(model)
    fused: List[List[str]] = []
    if fuse_map:
        fused.extend(list(map(list, fuse_map)))
    fused.extend(_collect_fuse_candidates(target))

    if fused:
        torch.ao.quantization.fuse_modules(target, fused, inplace=True)
    return target


# -------- Dynamic Quant (быстро и просто, но обычно слабее для Conv) --------
def dynamic_quantize_linear(
    model: nn.Module,
    dtype: torch.dtype = torch.qint8,
    exclude_name_patterns: Optional[List[str]] = None,
    modules: Optional[Sequence[type[nn.Module]]] = None,
) -> nn.Module:
    """Apply dynamic quantization to selected ``Linear``-like modules.

    Parameters
    ----------
    modules:
        Optional explicit module types to quantize. Defaults to detected ``nn.Linear`` subclasses.
    """

    from torch.ao.quantization import quantize_dynamic

    exclude_name_patterns = list(exclude_name_patterns or [])
    target_types: List[type[nn.Module]] = list(modules or [])
    if not target_types:
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and not name_is_excluded(name, exclude_name_patterns):
                if type(module) not in target_types:
                    target_types.append(type(module))
    if not target_types:
        target_types = [nn.Linear]

    qmodel = quantize_dynamic(model, qconfig_spec=set(target_types), dtype=dtype)
    return qmodel


# -------- Static PTQ INT8 (рекомендуется для CPU) --------
def ptq_static_int8(
    model: nn.Module,
    calib_loader: Iterable,
    *,
    backend: str = "x86",
    exclude_name_patterns: Optional[List[str]] = None,
    per_channel_weights: bool = True,
    num_calib_batches: int = 100,
    device: str = "cpu",
    example_input_size: Tuple[int, int, int, int] = (1, 3, 224, 224),
    fuse: bool = True,
    additional_fuse_map: Optional[Sequence[Sequence[str]]] = None,
) -> nn.Module:
    """Post-training static INT8 quantization using FX tooling."""

    if fuse:
        model = fuse_model(model, fuse_map=additional_fuse_map, inplace=True)

    torch.backends.quantized.engine = backend
    base_qconfig = get_default_qconfig(backend)
    weight_observer = (
        PerChannelMinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_channel_symmetric)
        if per_channel_weights
        else MinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_tensor_symmetric)
    )
    default_qconfig = QConfig(activation=base_qconfig.activation, weight=weight_observer)

    qconfig_mapping = (
        QConfigMapping()
        .set_global(default_qconfig)
        .set_module_name(".*Embedding.*", None)
        .set_module_name(".*Norm.*", None)
    )

    exclude_name_patterns = list(exclude_name_patterns or [])
    for pat in (r"\.lora_",):
        if pat not in exclude_name_patterns:
            exclude_name_patterns.append(pat)

    if exclude_name_patterns:
        for name, _ in model.named_modules():
            if name_is_excluded(name, exclude_name_patterns):
                qconfig_mapping = qconfig_mapping.set_module_name(name, None)

    model.eval().to(device)
    example_inputs = torch.randn(*example_input_size, device=device)
    prepared = prepare_fx(model, qconfig_mapping, example_inputs=example_inputs)

    prepared.eval()
    n = 0
    with torch.no_grad():
        for batch in calib_loader:
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            x = x.to(device)
            _ = prepared(x)
            n += 1
            if n >= num_calib_batches:
                break

    quantized = convert_fx(prepared)
    quantized.eval().to("cpu")
    return quantized


# -------- QAT (скелет, на будущее) --------
def qat_int8_skeleton(
    model: nn.Module,
    *,
    backend: str = "x86",
    device: str = "cpu",
    example_input_size: Tuple[int, int, int, int] = (1, 3, 224, 224),
) -> nn.Module:
    """Skeleton for quantization-aware training."""

    from torch.ao.quantization import get_default_qat_qconfig

    torch.backends.quantized.engine = backend
    qconfig_mapping = QConfigMapping().set_global(get_default_qat_qconfig(backend))
    model.train().to(device)
    example_inputs = torch.randn(*example_input_size, device=device)
    prepared = torch.ao.quantization.prepare_qat_fx(model, qconfig_mapping, example_inputs=example_inputs)
    prepared.eval()
    quantized = convert_fx(prepared)
    quantized.eval().to("cpu")
    return quantized


def _convert_model_dtype(
    model: nn.Module,
    dtype: torch.dtype,
    *,
    modules: Optional[Sequence[type[nn.Module]]] = None,
    inplace: bool = False,
    device: Optional[str] = None,
) -> nn.Module:
    target = model if inplace else copy.deepcopy(model)
    module_types = tuple(modules) if modules else (nn.Linear, nn.Conv2d)
    for module in target.modules():
        if isinstance(module, module_types):
            module.to(dtype=dtype)
    if device is not None:
        target.to(device)
    target.eval()
    return target


def quantize_fp16(
    model: nn.Module,
    *,
    modules: Optional[Sequence[type[nn.Module]]] = None,
    inplace: bool = False,
    device: Optional[str] = None,
) -> nn.Module:
    """Convert selected modules to ``float16`` for lightweight weight-only quantization."""

    return _convert_model_dtype(model, torch.float16, modules=modules, inplace=inplace, device=device)


def quantize_bf16(
    model: nn.Module,
    *,
    modules: Optional[Sequence[type[nn.Module]]] = None,
    inplace: bool = False,
    device: Optional[str] = None,
) -> nn.Module:
    """Convert selected modules to ``bfloat16`` precision."""

    return _convert_model_dtype(model, torch.bfloat16, modules=modules, inplace=inplace, device=device)


def apply_smooth_quant(
    model: nn.Module,
    calib_loader: Iterable,
    *,
    alpha: float = 0.5,
    max_batches: int = 64,
    device: str = "cpu",
    modules: Optional[Sequence[type[nn.Module]]] = None,
) -> nn.Module:
    """Apply SmoothQuant-style rescaling prior to INT8 PTQ flows.

    The method estimates per-channel activation statistics on ``calib_loader``
    and rescales weights/activations accordingly. Downstream PTQ (for example
    :func:`ptq_static_int8`) can then leverage the more uniform ranges to reduce
    quantization error.
    """

    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be within [0, 1]")

    target_types: Tuple[type[nn.Module], ...] = tuple(modules) if modules else (nn.Linear, nn.Conv2d)
    eps = 1e-8

    model.eval().to(device)

    activation_max: Dict[nn.Module, torch.Tensor] = {}
    handles = []

    def _make_hook(module: nn.Module):
        def hook(_: nn.Module, inputs: Tuple[torch.Tensor, ...], __: torch.Tensor | Tuple[torch.Tensor, ...]):
            if not inputs:
                return
            x = inputs[0]
            if not isinstance(x, torch.Tensor):
                return
            reduce_dims = tuple(d for d in range(x.ndim) if d != 1)
            stats = x.detach().abs().amax(dim=reduce_dims)
            if module in activation_max:
                activation_max[module] = torch.maximum(activation_max[module], stats)
            else:
                activation_max[module] = stats

        return hook

    for module in model.modules():
        if isinstance(module, target_types):
            if hasattr(module, "_smooth_quant_pre_handle"):
                handle = getattr(module, "_smooth_quant_pre_handle")
                if handle is not None:
                    handle.remove()
            handles.append(module.register_forward_hook(_make_hook(module)))

    consumed = 0
    with torch.no_grad():
        for batch in calib_loader:
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            if isinstance(x, torch.Tensor):
                _ = model(x.to(device))
            consumed += 1
            if consumed >= max_batches:
                break

    for handle in handles:
        handle.remove()

    for module in model.modules():
        if module not in activation_max or not isinstance(module, target_types):
            continue

        weight = module.weight.detach()
        if isinstance(module, nn.Linear):
            weight_stats = weight.abs().amax(dim=0)
            view_shape = (1, -1)
        else:
            weight_stats = weight.abs().amax(dim=(0, 2, 3))
            view_shape = (1, -1, 1, 1)

        act_stats = activation_max[module]
        if act_stats.ndim == 0:
            act_stats = act_stats.unsqueeze(0)

        weight_stats = torch.clamp(weight_stats, min=eps)
        act_stats = torch.clamp(act_stats.to(weight_stats.device), min=eps)

        scale = (act_stats ** alpha) / (weight_stats ** (1 - alpha))
        scale = torch.clamp(scale, min=eps)

        module.register_buffer("smooth_quant_scale", scale)

        weight_scale = scale.view(view_shape).to(weight.device)
        module.weight.data = module.weight.data * weight_scale

        def _pre_hook(mod: nn.Module, inputs: Tuple[torch.Tensor, ...]):
            if not inputs:
                return None
            x = inputs[0]
            if not isinstance(x, torch.Tensor):
                return None
            reshape = (1, -1) if isinstance(mod, nn.Linear) else (1, -1, 1, 1)
            return (x / scale.view(reshape).to(x.device),)

        handle = module.register_forward_pre_hook(_pre_hook)
        setattr(module, "_smooth_quant_pre_handle", handle)

    model.to("cpu")
    return model
