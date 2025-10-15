import torch
import torch.nn as nn
import torch_pruning as tp
import time

def measure_latency(model, device="cpu", iters=50, warmup=10, input_size=(1,3,224,224)):
    model.eval().to(device)
    x = torch.randn(*input_size, device=device)
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x)
    t0 = time.time()
    with torch.no_grad():
        for _ in range(iters):
            _ = model(x)
    t1 = time.time()
    return (t1 - t0)/iters
