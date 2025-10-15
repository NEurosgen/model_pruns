# Optimization/src/pruning/channel_pruning.py  (v2)
from __future__ import annotations
import re
from typing import Iterable, List, Optional, Dict
import torch
import torch.nn as nn
import torch_pruning as tp  # v2.x

def is_depthwise(conv: nn.Conv2d) -> bool:
    return isinstance(conv, nn.Conv2d) and conv.groups == conv.in_channels == conv.out_channels

def is_pointwise(conv: nn.Conv2d) -> bool:
    return isinstance(conv, nn.Conv2d) and conv.kernel_size == (1, 1) and conv.groups == 1

def name_is_excluded(name: str, patterns: Iterable[str]) -> bool:
    return any(re.search(p, name) for p in patterns) if patterns else False



def collect_ignored_layers(model: nn.Module, exclude_name_patterns: List[str]) -> List[nn.Module]:
    ignored = []
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d) and is_depthwise(m):
            ignored.append(m)
            continue
        if name_is_excluded(name, exclude_name_patterns):
            ignored.append(m)
    return ignored

def prune_model_channels(
    model: nn.Module,
    example_inputs: torch.Tensor,
    amount: float = 0.30,                              # желаемая доля каналов на слой
    *,
    exclude_name_patterns: Optional[List[str]] = None, # напр.: [r"lora_", r"^model\.classifier", r"\.base$"]
    min_out_channels: int = 8,
    importance: str = "l1",                            # "l1" | "l2"
    device: str = "cpu",
) -> nn.Module:
    model.eval().to(device)
    example_inputs = example_inputs.to(device)

    # включаем градиенты на построение внутреннего графа (v2 это нужно)
    prev = torch.is_grad_enabled()
    torch.set_grad_enabled(True)
    imp = tp.importance.MagnitudeImportance(p=(1 if importance == "l1" else 2))

    # дефолтные исключения: LoRA, classifier, .base-обёртки
    exclude_name_patterns = list(exclude_name_patterns or [])
    for pat in (r"\.base$", r"lora_", r"^model\.classifier"):
        if pat not in exclude_name_patterns:
            exclude_name_patterns.append(pat)

    ignored_layers = collect_ignored_layers(model, exclude_name_patterns)

    # Собираем список prunable Conv2d (groups=1, не depthwise, не исключён)
    prunable: List[nn.Conv2d] = []
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d) and m.groups == 1 and not is_depthwise(m):
            if not name_is_excluded(name, exclude_name_patterns):
                prunable.append(m)

    if not prunable:
        print("[channel_pruner] No eligible Conv2d modules to prune.")
        return model

    # Пер-слойная sparsity с учётом min_out_channels:
    ch_sparsity_dict: Dict[nn.Module, float] = {}
    for m in prunable:
        ch = m.out_channels
        # максимальная допустимая доля зануления, чтобы остаться >= min_out_channels
        max_sparsity = max(0.0, 1.0 - (min_out_channels / float(ch)))
        target = min(amount, max_sparsity)
        if target <= 0.0:
            continue
        ch_sparsity_dict[m] = float(target)

    if not ch_sparsity_dict:
        print("[channel_pruner] Nothing to prune given min_out_channels constraints.")
        return model

    # Можно слегка приоритизировать 1x1 conv через важность: в v2 глобальный режим сам «распределит»,
    # но мы используем per-layer dict, так что приоритет задаём самим подбором amount.
    pruner = tp.pruner.MagnitudePruner(
        model,
        example_inputs=example_inputs,
        importance=imp,
        global_pruning=False,               # per-layer
        ch_sparsity=None,                   # игнорируем; используем dict
        ch_sparsity_dict=ch_sparsity_dict,  # << ключевой момент
        ignored_layers=ignored_layers,
        root_module_types=[nn.Conv2d],
    )
    print("we are here")
    pruner.step()  # применяем один шаг прунинга (физически меняет слои и их зависимости)
 

    torch.set_grad_enabled(prev)

    model.to("cpu").eval()
    return model
