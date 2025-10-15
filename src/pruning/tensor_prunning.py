import torch
import torch.nn.utils.prune as prune
import torch.nn as nn
from collections import defaultdict
import time



def global_unstructured_prune(model, amount=0.3):
    """
    Globally prune the smallest-magnitude weights across Conv2d and Linear.
    This creates binary masks; weights stay same shape (no real speedup).
    """
    parameters_to_prune = []
    for name,module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            if "lora_" in name or "classifier" in name:
                continue
            parameters_to_prune.append((module, "weight"))

    # Global L1 unstructured
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=amount,   # e.g., 0.3 -> 30% of all weights zeroed globally
    )

    # (optional) prune biases a bit as well
    # for m in model.modules():
    #     if isinstance(m, (nn.Conv2d, nn.Linear)) and m.bias is not None:
    #         prune.l1_unstructured(m, name="bias", amount=amount/2)



def make_pruning_permanent(model):
    # Remove reparametrization so masks are baked into .weight (zeros stay).
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            if hasattr(m, "weight_mask"):
                prune.remove(m, "weight")
            if hasattr(m, "bias_mask"):
                prune.remove(m, "bias")

