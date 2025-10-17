"""
Современная квантизация с использованием PT2E API (PyTorch 2.0+)
Все функции переписаны с устаревшего FX подхода на новый PT2E (PyTorch 2 Export)
"""
from __future__ import annotations

import copy
import re
import time
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

__all__ = [
    "name_is_excluded",
    "measure_latency",
    "dynamic_quantize_linear",
    "ptq_static_int8",
    "qat_int8_prepare",
    "qat_int8_convert",
    "fuse_model",
    "quantize_fp16",
    "quantize_bf16",
    "apply_smooth_quant",
]


# ============================================================================
# УТИЛИТЫ
# ============================================================================

def name_is_excluded(name: str, patterns: Iterable[str]) -> bool:
    """Проверяет, совпадает ли имя с любым из regex паттернов."""
    return any(re.search(p, name) for p in (patterns or []))


@torch.no_grad()
def measure_latency(
    model: nn.Module,
    input_size: Tuple[int, int, int, int] = (1, 3, 224, 224),
    iters: int = 30,
    warmup: int = 10,
    device: str = "cpu",
) -> float:
    """Измеряет среднюю латентность модели (в секундах)."""
    model.eval().to(device)
    x = torch.randn(*input_size, device=device)
    
    # Прогрев
    for _ in range(warmup):
        _ = model(x)
    
    # Измерение
    t0 = time.time()
    for _ in range(iters):
        _ = model(x)
    t1 = time.time()
    
    return (t1 - t0) / max(1, iters)


def _collect_fuse_candidates(module: nn.Module, prefix: str = "") -> List[List[str]]:
    """
    Эвристически находит паттерны Conv2d+BatchNorm(+ReLU) для слияния.
    Fusion улучшает скорость и точность квантизации.
    """

    def _from_sequential(seq_module: nn.Sequential, seq_prefix: str) -> List[List[str]]:
        """Собирает последовательности для слияния внутри nn.Sequential."""
        sequential_fuse: List[List[str]] = []
        sub_names = list(seq_module._modules.keys())
        idx = 0

        while idx < len(sub_names) - 1:
            first_name = sub_names[idx]
            second_name = sub_names[idx + 1]
            first = seq_module._modules[first_name]
            second = seq_module._modules[second_name]

            if isinstance(first, nn.Conv2d) and isinstance(second, nn.BatchNorm2d):
                # Проверяем, есть ли ReLU после
                if idx + 2 < len(sub_names):
                    third_name = sub_names[idx + 2]
                    third = seq_module._modules[third_name]
                    if isinstance(third, (nn.ReLU, nn.ReLU6)):
                        sequential_fuse.append([
                            f"{seq_prefix}{first_name}",
                            f"{seq_prefix}{second_name}",
                            f"{seq_prefix}{third_name}",
                        ])
                        idx += 3
                        continue

                # Conv+BN без ReLU
                sequential_fuse.append([
                    f"{seq_prefix}{first_name}",
                    f"{seq_prefix}{second_name}",
                ])
                idx += 2
                continue

            idx += 1

        return sequential_fuse

    fuse_list: List[List[str]] = []

    if isinstance(module, nn.Sequential):
        # Обрабатываем сам модуль, чтобы корректно работать и для корневых nn.Sequential
        fuse_list.extend(_from_sequential(module, prefix))

    for name, child in module.named_children():
        child_prefix = f"{prefix}{name}."
        fuse_list.extend(_collect_fuse_candidates(child, prefix=child_prefix))

    return fuse_list


def fuse_model(
    model: nn.Module,
    fuse_map: Optional[Sequence[Sequence[str]]] = None,
    *,
    inplace: bool = True,
) -> nn.Module:
    """
    Сливает соседние слои Conv-BN-(ReLU) перед квантизацией.
    
    Зачем: BatchNorm можно "впитать" в Conv, что упрощает граф и улучшает квантизацию.
    
    Args:
        model: Исходная модель
        fuse_map: Дополнительные группы слоёв для слияния
        inplace: Изменять исходную модель или создать копию
    """
    target = model if inplace else copy.deepcopy(model)
    fused: List[List[str]] = []
    
    if fuse_map:
        fused.extend(list(map(list, fuse_map)))
    
    # Автоматический поиск паттернов
    fused.extend(_collect_fuse_candidates(target))
    
    if fused:
        torch.ao.quantization.fuse_modules(target, fused, inplace=True)
    
    return target


# ============================================================================
# DYNAMIC QUANTIZATION (для Linear слоёв, без калибровки)
# ============================================================================

def dynamic_quantize_linear(
    model: nn.Module,
    dtype: torch.dtype = torch.qint8,
    exclude_name_patterns: Optional[List[str]] = None,
    modules: Optional[Sequence[type[nn.Module]]] = None,
) -> nn.Module:
    """
    Динамическая квантизация Linear слоёв (современный API, работает на PyTorch 1.13+).
    
    Как работает:
    - Веса квантизуются в int8 один раз при загрузке
    - Активации квантизуются динамически при каждом forward pass
    - Не требует калибровки!
    
    Применение: NLP модели (BERT, GPT), где большинство вычислений в Linear слоях.
    """
    from torch.ao.quantization import quantize_dynamic
    
    exclude_name_patterns = list(exclude_name_patterns or [])
    target_types: List[type[nn.Module]] = list(modules or [])
    
    # Если типы не указаны, автоматически находим все Linear слои
    if not target_types:
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and not name_is_excluded(name, exclude_name_patterns):
                if type(module) not in target_types:
                    target_types.append(type(module))
    
    if not target_types:
        target_types = [nn.Linear]
    
    # quantize_dynamic - стабильный API, не deprecated
    qmodel = quantize_dynamic(model, qconfig_spec=set(target_types), dtype=dtype)
    return qmodel


# ============================================================================
# STATIC POST-TRAINING QUANTIZATION (PTQ) - INT8
# ============================================================================

def ptq_static_int8(
    model: nn.Module,
    calib_loader: Iterable,
    *,
    device: str = "cpu",
    example_input_size: Tuple[int, int, int, int] = (1, 3, 224, 224),
    num_calib_batches: int = 100,
    per_channel_weights: bool = True,
) -> nn.Module:
    """
    Статическая пост-тренировочная квантизация в INT8 (PT2E flow).
    
    Этапы:
    1. torch.export.export() - экспортируем модель в стабильный граф
    2. prepare_pt2e() - вставляем observers для измерения статистики
    3. Калибровка - прогоняем данные, observers собирают min/max
    4. convert_pt2e() - заменяем операции на квантизованные
    
    Args:
        model: Модель для квантизации
        calib_loader: DataLoader с калибровочными данными
        device: Устройство для калибровки
        example_input_size: Размер входа для экспорта
        num_calib_batches: Сколько батчей использовать для калибровки
        per_channel_weights: True = per-channel (точнее), False = per-tensor (быстрее)
    
    Returns:
        Квантизованная модель (callable nn.Module)
    """
    from torch.ao.quantization.quantize_pt2e import prepare_pt2e, convert_pt2e
    from torch.ao.quantization.quantizer.xnnpack_quantizer import (
        XNNPACKQuantizer,
        get_symmetric_quantization_config,
    )
    
    model = model.eval().to(device)
    example_inputs = (torch.randn(*example_input_size, device=device),)
    
    # 1. Экспорт модели (torch.export требует PyTorch 2.1+)
    print("Экспортируем модель...")
    exported = torch.export.export(model, example_inputs)
    
    # 2. Настраиваем квантизатор
    quantizer = XNNPACKQuantizer()
    
    # Симметричная квантизация: [-127, 127] для весов и активаций
    # per_channel=True означает свой scale для каждого выходного канала Conv/Linear
    quantizer.set_global(
        get_symmetric_quantization_config(is_per_channel=per_channel_weights)
    )
    
    # 3. Prepare - вставляем observers
    print("Подготавливаем модель (вставляем observers)...")
    prepared = prepare_pt2e(exported, quantizer)
    
    # 4. Калибровка - прогоняем данные для сбора статистики
    print(f"Калибровка на {num_calib_batches} батчах...")
    with torch.no_grad():
        n = 0
        for batch in calib_loader:
            # Извлекаем входные данные (обрабатываем разные форматы DataLoader)
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            x = x.to(device)
            
            # prepared - это обёртка ExportedProgram, вызываем .module()
            prepared.module()(x)
            
            n += 1
            if n >= num_calib_batches:
                break
    
    # 5. Convert - заменяем операции на квантизованные
    print("Конвертируем в квантизованную модель...")
    quantized = convert_pt2e(prepared)
    
    # Возвращаем обычный nn.Module
    quantized_module = quantized.module().eval().to("cpu")
    return quantized_module


# ============================================================================
# QUANTIZATION-AWARE TRAINING (QAT) - INT8
# ============================================================================

def qat_int8_prepare(
    model: nn.Module,
    *,
    device: str = "cpu",
    example_input_size: Tuple[int, int, int, int] = (1, 3, 224, 224),
    per_channel_weights: bool = True,
) -> nn.Module:
    """
    Подготавливает модель для QAT (Quantization-Aware Training).
    
    QAT vs PTQ:
    - PTQ: квантизуем уже обученную модель → быстро, но может потерять точность
    - QAT: обучаем модель с симуляцией квантизации → дольше, но точнее
    
    Как работает:
    - В forward pass: веса и активации проходят через fake_quantize (симуляция квантизации)
    - В backward pass: градиенты проходят через STE (Straight-Through Estimator)
    - Модель учится быть устойчивой к ошибкам квантизации
    
    Использование:
    1. prepared = qat_int8_prepare(model)
    2. # Обучаем prepared как обычную модель
    3. quantized = qat_int8_convert(prepared)
    
    Returns:
        Подготовленная модель для обучения (в режиме .train())
    """
    from torch.ao.quantization.quantize_pt2e import prepare_qat_pt2e
    from torch.ao.quantization.quantizer.xnnpack_quantizer import (
        XNNPACKQuantizer,
        get_symmetric_quantization_config,
    )
    
    model = model.train().to(device)
    example_inputs = (torch.randn(*example_input_size, device=device),)
    
    print("Экспортируем модель для QAT...")
    exported = torch.export.export(model, example_inputs)
    
    quantizer = XNNPACKQuantizer()
    quantizer.set_global(
        get_symmetric_quantization_config(is_per_channel=per_channel_weights)
    )
    
    print("Подготавливаем модель для QAT (вставляем fake_quantize)...")
    prepared = prepare_qat_pt2e(exported, quantizer)
    
    # Возвращаем модель в режиме train
    return prepared.module().train().to(device)


def qat_int8_convert(
    prepared_model: nn.Module,
    *,
    device: str = "cpu",
    example_input_size: Tuple[int, int, int, int] = (1, 3, 224, 224),
) -> nn.Module:
    """
    Конвертирует QAT модель в финальную квантизованную версию.
    
    Вызывать после завершения обучения с qat_int8_prepare().
    
    Args:
        prepared_model: Модель после QAT обучения
        device: Устройство для экспорта
        example_input_size: Размер входа
    
    Returns:
        Финальная квантизованная модель для инференса
    """
    from torch.ao.quantization.quantize_pt2e import convert_pt2e
    
    prepared_model = prepared_model.eval().to(device)
    example_inputs = (torch.randn(*example_input_size, device=device),)
    
    print("Экспортируем обученную QAT модель...")
    exported = torch.export.export(prepared_model, example_inputs)
    
    print("Конвертируем QAT модель в квантизованную...")
    quantized = convert_pt2e(exported)
    
    return quantized.module().eval().to("cpu")


# ============================================================================
# FP16 / BF16 QUANTIZATION (простое приведение типов)
# ============================================================================

def _convert_model_dtype(
    model: nn.Module,
    dtype: torch.dtype,
    *,
    modules: Optional[Sequence[type[nn.Module]]] = None,
    inplace: bool = False,
    device: Optional[str] = None,
) -> nn.Module:
    """Конвертирует параметры модели в указанный dtype."""
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
    """
    Конвертирует модель в float16 (лёгкая квантизация весов).
    
    Плюсы: очень простая, почти без потери точности, 2x экономия памяти
    Минусы: не даёт такого ускорения как int8, нужна GPU поддержка FP16
    """
    return _convert_model_dtype(
        model, torch.float16, modules=modules, inplace=inplace, device=device
    )


def quantize_bf16(
    model: nn.Module,
    *,
    modules: Optional[Sequence[type[nn.Module]]] = None,
    inplace: bool = False,
    device: Optional[str] = None,
) -> nn.Module:
    """
    Конвертирует модель в bfloat16.
    
    BF16 vs FP16:
    - BF16: тот же диапазон что у FP32, но меньше точность мантиссы
    - FP16: больше точность мантиссы, но меньше диапазон → может overflow
    - BF16 обычно стабильнее для обучения
    """
    return _convert_model_dtype(
        model, torch.bfloat16, modules=modules, inplace=inplace, device=device
    )


# ============================================================================
# SMOOTHQUANT - продвинутая техника для улучшения PTQ
# ============================================================================

def apply_smooth_quant(
    model: nn.Module,
    calib_loader: Iterable,
    *,
    alpha: float = 0.5,
    max_batches: int = 64,
    device: str = "cpu",
    modules: Optional[Sequence[type[nn.Module]]] = None,
) -> nn.Module:
    """
    Применяет SmoothQuant - rescaling весов/активаций перед INT8 PTQ.
    
    Проблема: активации часто имеют outliers (выбросы), которые плохо квантизуются
    Решение: перераспределяем сложность между весами и активациями
    
    Как работает:
    1. Измеряем max активаций по каналам: X_max
    2. Измеряем max весов по каналам: W_max
    3. Вычисляем scale = (X_max^α) / (W_max^(1-α))
    4. Веса умножаем на scale, активации делим на scale
    5. Теперь и веса, и активации квантизуются лучше!
    
    Args:
        alpha: баланс между весами и активациями (0.5 = равномерно)
               0.0 = не трогаем веса, только активации
               1.0 = не трогаем активации, только веса
    
    Применять ПЕРЕД ptq_static_int8()!
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha должен быть в [0, 1]")
    
    target_types: Tuple[type[nn.Module], ...] = tuple(modules) if modules else (nn.Linear, nn.Conv2d)
    eps = 1e-8
    
    model.eval().to(device)
    
    # Словарь для хранения максимальных активаций
    activation_max: Dict[nn.Module, torch.Tensor] = {}
    handles = []
    
    def _make_hook(module: nn.Module):
        """Создаёт hook для измерения активаций."""
        def hook(_: nn.Module, inputs: Tuple[torch.Tensor, ...], __):
            if not inputs:
                return
            x = inputs[0]
            if not isinstance(x, torch.Tensor):
                return
            
            # Вычисляем max по всем измерениям кроме channel (dim=1)
            reduce_dims = tuple(d for d in range(x.ndim) if d != 1)
            stats = x.detach().abs().amax(dim=reduce_dims)
            
            # Обновляем максимум
            if module in activation_max:
                activation_max[module] = torch.maximum(activation_max[module], stats)
            else:
                activation_max[module] = stats
        
        return hook
    
    # Регистрируем hooks для всех целевых слоёв
    for module in model.modules():
        if isinstance(module, target_types):
            handles.append(module.register_forward_hook(_make_hook(module)))
    
    # Прогоняем калибровочные данные
    print(f"SmoothQuant: собираем статистику активаций на {max_batches} батчах...")
    consumed = 0
    with torch.no_grad():
        for batch in calib_loader:
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            if isinstance(x, torch.Tensor):
                _ = model(x.to(device))
            consumed += 1
            if consumed >= max_batches:
                break
    
    # Удаляем hooks
    for handle in handles:
        handle.remove()
    
    # Применяем rescaling к каждому слою
    print("SmoothQuant: применяем rescaling...")
    for module in model.modules():
        if module not in activation_max or not isinstance(module, target_types):
            continue
        
        weight = module.weight.detach()
        
        # Вычисляем статистику весов
        if isinstance(module, nn.Linear):
            # Linear: [out_features, in_features]
            weight_stats = weight.abs().amax(dim=0)  # max по выходным каналам
            view_shape = (1, -1)  # для Broadcasting
        else:
            # Conv2d: [out_channels, in_channels, H, W]
            weight_stats = weight.abs().amax(dim=(0, 2, 3))  # max по out, H, W
            view_shape = (1, -1, 1, 1)
        
        act_stats = activation_max[module]
        if act_stats.ndim == 0:
            act_stats = act_stats.unsqueeze(0)
        
        # Избегаем деления на ноль
        weight_stats = torch.clamp(weight_stats, min=eps)
        act_stats = torch.clamp(act_stats.to(weight_stats.device), min=eps)
        
        # Формула SmoothQuant: scale = (X_max^α) / (W_max^(1-α))
        scale = (act_stats ** alpha) / (weight_stats ** (1 - alpha))
        scale = torch.clamp(scale, min=eps)
        
        # Сохраняем scale как buffer (не будет обучаться)
        module.register_buffer("smooth_quant_scale", scale)
        
        # Умножаем веса на scale
        weight_scale = scale.view(view_shape).to(weight.device)
        module.weight.data = module.weight.data * weight_scale
        
        # Регистрируем pre_hook для деления активаций на scale
        def _pre_hook(mod: nn.Module, inputs: Tuple[torch.Tensor, ...]):
            if not inputs:
                return None
            x = inputs[0]
            if not isinstance(x, torch.Tensor):
                return None
            
            # Берём scale из buffer
            scale = mod.smooth_quant_scale
            reshape = (1, -1) if isinstance(mod, nn.Linear) else (1, -1, 1, 1)
            return (x / scale.view(reshape).to(x.device),)
        
        handle = module.register_forward_pre_hook(_pre_hook)
        setattr(module, "_smooth_quant_pre_handle", handle)
    
    model.to("cpu")
    print("SmoothQuant применён! Теперь можно использовать ptq_static_int8()")
    return model