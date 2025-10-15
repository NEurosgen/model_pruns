
import torch
import torch.nn.utils.prune as prune
import torch.nn as nn


def report_sparsity(model):
    total_zeros, total_params = 0, 0
    per_layer = []
    for name, m in model.named_modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            w = m.weight.data
            zeros = torch.sum(w == 0).item()
            params = w.numel()
            total_zeros += zeros
            total_params += params
            per_layer.append((name, zeros/params if params else 0.0))
    print("Global sparsity: {:.2f}%".format(100 * total_zeros / max(1, total_params)))
    for name, sp in sorted(per_layer, key=lambda t: -t[1])[:8]:
        print(f"  {name:40s} {100*sp:6.2f}%")