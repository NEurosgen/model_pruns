# main_prune_mnv3_fixed.py
# -*- coding: utf-8 -*-

import sys
from typing import Tuple, Optional
from ..checkpoints_utils.checkpoint_load import load_model_from_checkpoint
import torch
import torch.nn as nn
import torchvision
from torchvision.models.mobilenetv3 import mobilenet_v3_small, InvertedResidual
import torch_pruning as tp

from ..report_utils.measure_latency import measure_latency
from ..report_utils.performence_compare import model_performance


import torch
from torch.ao.quantization import get_default_qat_qconfig
from torch.ao.quantization.quantize_fx import QConfigMapping, convert_fx, prepare_qat_fx
import torch
import torch.nn as nn
from typing import Optional, Tuple

def _get_scale(mod) -> float:
    # Популярные поля: lora_alpha, r, scaling. Подстрой под свою реализацию при необходимости.
    if hasattr(mod, "lora_alpha") and hasattr(mod, "r") and mod.r:
        return float(mod.lora_alpha) / float(mod.r)
    if hasattr(mod, "scaling"):
        return float(mod.scaling)
    return 1.0  # запасной вариант

@torch.no_grad()
def merge_lora_linear(mod: nn.Module) -> nn.Linear:
    """
    Сливает LoRA в базовый nn.Linear.
    Требования:
      - mod.base: nn.Linear (обязателен)
      - mod.lora_A: nn.Linear (r, in_features)
      - mod.lora_B: nn.Linear (out_features, r)
    Возврат: новый nn.Linear с W_eff и прежним bias.
    """
    assert isinstance(mod.base, nn.Linear), "mod.base должен быть nn.Linear"
    base: nn.Linear = mod.base
    A: nn.Linear = getattr(mod, "lora_A", None)
    B: nn.Linear = getattr(mod, "lora_B", None)
    if A is None or B is None:
        return base  # нечего сливать

    scale = _get_scale(mod)

    W = base.weight        # [out, in]
    W_eff = W
    if hasattr(B, "weight") and hasattr(A, "weight"):
        W_eff = W + scale * (B.weight @ A.weight)  # [out, r] @ [r, in] -> [out, in]

    new_linear = nn.Linear(base.in_features, base.out_features, bias=(base.bias is not None))
    new_linear.weight.copy_(W_eff)
    if base.bias is not None:
        new_linear.bias.copy_(base.bias)

    return new_linear

@torch.no_grad()
def merge_lora_conv1x1(mod: nn.Module) -> nn.Conv2d:
    """
    Сливает LoRA в базовый nn.Conv2d (для 1x1 свёрток, stride=1, padding=0, groups=1).
    Требования:
      - mod.base: nn.Conv2d(kernel_size=1, groups=1)
      - mod.lora_A: nn.Conv2d(r, in, 1,1)
      - mod.lora_B: nn.Conv2d(out, r, 1,1)
    Возврат: новый nn.Conv2d с весом W_eff и исходным bias.
    """
    assert isinstance(mod.base, nn.Conv2d), "mod.base должен быть nn.Conv2d"
    base: nn.Conv2d = mod.base
    A: nn.Conv2d = getattr(mod, "lora_A", None)
    B: nn.Conv2d = getattr(mod, "lora_B", None)
    if A is None or B is None:
        return base

    # Проверим, что это действительно 1x1 без групп и без смещения по геометрии.
    k = base.kernel_size
    assert k == (1, 1) and base.groups == 1 and base.stride == (1, 1) and base.padding == (0, 0), \
        "Этот merge реализован для 1x1 Conv2d с stride=1, padding=0, groups=1"

    scale = _get_scale(mod)

    W = base.weight  # [out, in, 1, 1]
    # Превращаем в матрицы и обратно
    Wb = B.weight.view(B.out_channels, B.in_channels)      # [out, r]
    Wa = A.weight.view(A.out_channels, A.in_channels)      # [r, in]
    W_eff = W + scale * (Wb @ Wa).view_as(W)

    new_conv = nn.Conv2d(
        in_channels=base.in_channels,
        out_channels=base.out_channels,
        kernel_size=1,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=(base.bias is not None),
    )
    new_conv.weight.copy_(W_eff)
    if base.bias is not None:
        new_conv.bias.copy_(base.bias)

    return new_conv

def _is_lora_wrapper(m: nn.Module) -> bool:
    return hasattr(m, "base") and hasattr(m, "lora_A") and hasattr(m, "lora_B")

from torchvision.models import mobilenet_v3_large

@torch.no_grad()
def merge_all_lora_inplace(module: nn.Module) -> None:
    """
    Находит все LoRA-обёртки в модели, сливает их и ЗАМЕНЯЕТ эти подмодули на базовые Conv/Linear.
    После этого в графе не останется вызовов lora_A/lora_B.
    """
    for name, child in list(module.named_children()):
        if _is_lora_wrapper(child):
            base = child.base
            device = base.weight.device
            dtype = base.weight.dtype

            if isinstance(base, nn.Linear):
                merged = merge_lora_linear(child).to(device=device, dtype=dtype)
            elif isinstance(base, nn.Conv2d):
                # Поддержан распространённый случай 1x1-конволюций.
                merged = merge_lora_conv1x1(child).to(device=device, dtype=dtype)
            else:
                raise NotImplementedError(f"LoRA merge не реализован для типа {type(base)}")

            # Заменяем модуль на слитый базовый
            setattr(module, name, merged)
        else:
            merge_all_lora_inplace(child)

# ==== Пример использования ====
# model = ...  # твоя MobileNetV3+LoRA после дообучения LoRA
# model.eval()
# merge_all_lora_inplace(model)
# # Теперь модель «чистая» — без lora_A/lora_B, можно делать PTQ/QAT и export.

def main():
    torch.set_float32_matmul_precision("high")
    device = "cpu"
    model = load_model_from_checkpoint("/home/temp/MyDir/Projects/aspdfpwjfpwejfwpefwef/tb_logs_big/mobilnet/version_3/checkpoints/epoch=0-step=88.ckpt")
    print("Time before merege:", measure_latency(model))
    merge_all_lora_inplace(model.model)
    print("Time:", measure_latency(model))
    example = (torch.randn(1,3,224,224),)
    #print(model)
    print("not error")
    model = model.model
    model.train()

    # === 2. Настройка под ARM (QNNPACK) ===
    torch.backends.quantized.engine = "qnnpack"
    qconfig = get_default_qat_qconfig("qnnpack")
    qmap = QConfigMapping().set_global(qconfig)  # 

    # === 3. Подготовка к QAT ===
    example = torch.randn(1, 3, 224, 224)
    prep = prepare_qat_fx(model, qconfig_mapping=qmap, example_inputs=example)

    prep.eval() 
    with torch.no_grad():
        for _ in range(32):
            _ = prep(torch.randn(1, 3, 224, 224))


    prep.eval()
    int8_model = convert_fx(prep.cpu())

    # === 6. Экспорт под мобильный рантайм ===
    example = torch.randn(1, 3, 224, 224)
    traced = torch.jit.trace(int8_model, example)
    traced = torch.jit.optimize_for_inference(traced)

    from torch.utils.mobile_optimizer import optimize_for_mobile
    lite = optimize_for_mobile(traced)
    print(int8_model)
    #print("Optimized Time:", measure_latency(lite))
    
    lite._save_for_lite_interpreter("mobilenetv3_qat_qnnpack.ptl")

if __name__ == "__main__":
    main()
