from ..checkpoints_utils.checkpoint_load import load_model_from_checkpoint
from  ..pruning.tensor_prunning  import global_unstructured_prune,make_pruning_permanent
from ..report_utils.measure_latency import measure_latency
from .. report_utils.report_sparsity import report_sparsity
from ..pruning.channel_pruning import prune_model_channels
from  omegaconf import OmegaConf
from ..pruning.channel_pruning import progressive_channel_pruning

import torch
from ..quantization.quant import (
    dynamic_quantize_linear,
    ptq_static_int8,
    qat_int8_prepare,
    qat_int8_convert,
    fuse_model,
    quantize_fp16,
    apply_smooth_quant,
    measure_latency,
)

cfg = OmegaConf.load("/home/temp/MyDir/Projects/aspdfpwjfpwejfwpefwef/Drons/tb_logs_big/efficientnet/version_6/hparams.yaml")


model = load_model_from_checkpoint(checkpoint_path="tb_logs_big/efficientnet/version_0/checkpoints/epoch=1-step=88.ckpt")

example_inputs = torch.randn(1, 3, 224, 224)

from ..pruning import tensor_prunning as tp

lat0 = measure_latency(model = model, input_size= (1,3,224,224))
# print(f"Latency before: {lat0:.6f}s")
# tp.global_unstructured_prune(model, amount=0.9)
# tp.layer_unstructured_prune(model, amount=0.9)
# tp.structured_filter_prune(model, amount=0.9, n=2, dim=0)

# # Случайный прунинг (полезен для бенчмарков)
# tp.random_unstructured_prune(model, amount=0.9)

# # Жёсткое обнуление весов по порогу
# threshold = 1
# tp.threshold_prune(model, threshold, prune_bias=True)

# # Отчёт по разреженности и фиксация масок
# stats = tp.report_sparsity(model)
# tp.make_pruning_permanent(model)
# print(f"After pruning: {measure_latency(model):.4f}s")

# print(stats)

size_before = sum(p.numel() * p.element_size() for p in model.parameters())
print(f"Размер (FP32): {size_before / 1024:.2f} KB")
prepared_model = qat_int8_prepare(
    model,
    device="cpu",

)
# Конвертация в FP16
model_fp16 = qat_int8_convert(model)
size_after = sum(p.numel() * p.element_size() for p in model_fp16.parameters())
print(f"Размер (FP16): {size_after / 1024:.2f} KB")
print(f"Сжатие: {size_before / size_after:.2f}x")


# calib_data = [torch.randn(1, 3, 224, 224) for _ in range(32)]
# apply_smooth_quant(model, calib_data, alpha=0.5, max_batches=32)
# quantized = ptq_static_int8(model, calib_data, num_calib_batches=32)

#print(f"INT8 latency: {measure_latency(quantized):.4f}s")