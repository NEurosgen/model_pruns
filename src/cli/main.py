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
from torch.ao.quantization.quantize_fx import QConfigMapping, prepare_fx, convert_fx,prepare_qat_fx
from ..report_utils.measure_latency import measure_latency
def main():
    torch.set_float32_matmul_precision("high")
    device = "cpu"
    model = load_model_from_checkpoint("/home/temp/MyDir/Projects/aspdfpwjfpwejfwpefwef/tb_logs_big/mobilnet/version_2/checkpoints/epoch=99-step=8800.ckpt").model
    size_before = sum(p.numel() * p.element_size() for p in model.parameters())
    model_qt = convert_fx(model)
    size_after = sum(p.numel() * p.element_size() for p in model_qt.parameters())
    print(f"Размер (FP16): {size_after / 1024:.2f} KB")
    print(f"Сжатие: {size_before / size_after:.2f}x")

    lat0 = measure_latency(model = model, input_size= (1,3,224,224))
    print(f"Latency before: {lat0:.6f}s")
if __name__ == "__main__":
    main()
