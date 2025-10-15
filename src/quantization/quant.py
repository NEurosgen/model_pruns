# Optimization/src/quant/weight_quant.py
from __future__ import annotations
from typing import Iterable, List, Optional, Tuple, Dict
import re
import torch
import torch.nn as nn
from torch.ao.quantization import (
    get_default_qconfig,
    QConfigMapping,
    prepare_fx,
    convert_fx,
)

# -------- utils --------
def name_is_excluded(name: str, patterns: Iterable[str]) -> bool:
    return any(re.search(p, name) for p in (patterns or []))

@torch.no_grad()
def measure_latency(model: nn.Module, input_size=(1,3,224,224), iters=30, warmup=10, device="cpu") -> float:
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

# -------- Dynamic Quant (быстро и просто, но обычно слабее для Conv) --------
def dynamic_quantize_linear(model: nn.Module, dtype=torch.qint8, exclude_name_patterns: Optional[List[str]] = None) -> nn.Module:
    """
    Применяет динамическую квантизацию к Linear-слоям (и LSTM, если бы были).
    Актуально для классификатора/MLP головы.
    """
    from torch.ao.quantization import quantize_dynamic
    modules = {}
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear) and not name_is_excluded(name, exclude_name_patterns or []):
            modules[type(m)] = modules.get(type(m), m.__class__)
    if not modules:
        # по умолчанию нацелимся на все Linear
        modules = {nn.Linear: nn.Linear}
    qmodel = quantize_dynamic(model, qconfig_spec=set(modules.keys()), dtype=dtype)
    return qmodel

# -------- Static PTQ INT8 (рекомендуется для CPU) --------
def ptq_static_int8(
    model: nn.Module,
    calib_loader,                           # DataLoader или iterable из (x, y) / (x,)
    *,
    backend: str = "x86",                   # "x86" (OneDNN) для Intel/AMD, "qnnpack" для ARM
    exclude_name_patterns: Optional[List[str]] = None,
    per_channel_weights: bool = True,       # лучше для Conv (повышает качество)
    num_calib_batches: int = 100,           # хватит 100-300 батчей
    device: str = "cpu",
    example_input_size: Tuple[int,int,int,int] = (1,3,224,224),
) -> nn.Module:
    """
    Post-Training Static Quantization (FX).
    Квантует Conv/Linear и активации (INT8) с калибровкой на небольшом датасете.
    """
    # 1) Бэкенд
    torch.backends.quantized.engine = backend

    # 2) QConfig (по умолчанию x86 -> OneDNN): уже включает разумные схемы, поддерживает per-channel для Conv
    default_qconfig = get_default_qconfig(backend)

    # 3) QConfigMapping: можно тонко исключать типы/имена
    #    По умолчанию квантуем всё подходящее; Embedding и нормализации оставляем FP32.
    qconfig_mapping = QConfigMapping().set_global(default_qconfig) \
                                      .set_module_name(".*Embedding.*", None) \
                                      .set_module_name(".*Norm.*", None)

    # 3a) Исключить по именам (regex), например LoRA, classifier и т.д.
    exclude_name_patterns = list(exclude_name_patterns or [])
    for pat in (r"\.lora_",):  # мягкое исключение LoRA-подмодулей
        if pat not in exclude_name_patterns:
            exclude_name_patterns.append(pat)

    if exclude_name_patterns:
        for name, _ in model.named_modules():
            if name_is_excluded(name, exclude_name_patterns):
                qconfig_mapping = qconfig_mapping.set_module_name(name, None)

    # 4) Подготовка к квантизации (FX-prepare) — вставляет observer'ы
    model.eval().to(device)
    example_inputs = torch.randn(*example_input_size, device=device)
    prepared = prepare_fx(model, qconfig_mapping, example_inputs=example_inputs)

    # 5) Калибровка: просто прогоняем несколько батчей без градиентов
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

    # 6) Конвертация: вставляет квант/деквант и заменяет модули на INT8
    quantized = convert_fx(prepared)

    # 7) Готово: quantized — это модель с INT8 весами/активациями (для поддерживаемых слоёв)
    quantized.eval().to("cpu")
    return quantized

# -------- QAT (скелет, на будущее) --------
def qat_int8_skeleton(
    model: nn.Module,
    *,
    backend: str = "x86",
    device: str = "cpu",
    example_input_size=(1,3,224,224),
) -> nn.Module:
    """
    Скелет для QAT (обучение с имитацией INT8). Нужен отдельный цикл тренировки.
    Оставлен как заготовка — обычно PTQ достаточно.
    """
    from torch.ao.quantization import get_default_qat_qconfig
    torch.backends.quantized.engine = backend
    qconfig_mapping = QConfigMapping().set_global(get_default_qat_qconfig(backend))
    model.train().to(device)
    example_inputs = torch.randn(*example_input_size, device=device)
    prepared = torch.ao.quantization.prepare_qat_fx(model, qconfig_mapping, example_inputs=example_inputs)
    # --- тут вы обучаете prepared как обычно несколько эпох ---
    # после обучения:
    prepared.eval()
    quantized = convert_fx(prepared)
    quantized.eval().to("cpu")
    return quantized
