import pytest
import torch
import torch.nn as nn

from Optimization.src.quantization import quant


def test_name_is_excluded_matches_regex():
    assert quant.name_is_excluded("layer.lora_adapter", ["lora_"])
    assert not quant.name_is_excluded("layer.weight", ["bias"])


class LatencyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4)

    def forward(self, x):
        return self.linear(x)


def test_measure_latency_runs(monkeypatch):
    model = LatencyNet()

    timestamps = iter([0.0, 0.0, 0.01])
    monkeypatch.setattr(quant.time, "time", lambda: next(timestamps))

    latency = quant.measure_latency(model, input_size=(1, 4), iters=1, warmup=0)
    assert latency == pytest.approx(0.01)


def test_dynamic_quantize_linear_replaces_layers():
    model = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 2))
    qmodel = quant.dynamic_quantize_linear(model)
    quantized_layers = [m for m in qmodel.modules() if m.__class__.__name__ == "Linear"]
    assert quantized_layers
    assert all(m.__class__.__module__.startswith("torch.nn.quantized.dynamic") for m in quantized_layers)


class FuseNet(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Conv2d(3, 3, kernel_size=3, bias=False),
            nn.BatchNorm2d(3),
            nn.ReLU(),
            nn.Conv2d(3, 3, kernel_size=3, bias=False),
            nn.BatchNorm2d(3),
        )


def test_fuse_model_auto_detects_patterns():
    model = FuseNet()
    fused = quant.fuse_model(model, inplace=False)
    # После слияния первый блок заменяется fused ConvBNReLU
    assert fused[0].__class__.__module__.startswith("torch.nn.intrinsic")


def _collect_dtypes(model):
    return [module.weight.dtype for module in model.modules() if hasattr(module, "weight")]


def test_quantize_fp16_and_bf16_change_dtype():
    model = LatencyNet()
    fp16_model = quant.quantize_fp16(model, inplace=False)
    bf16_model = quant.quantize_bf16(model, inplace=False)
    assert all(dtype == torch.float16 for dtype in _collect_dtypes(fp16_model))
    assert all(dtype == torch.bfloat16 for dtype in _collect_dtypes(bf16_model))
    # Исходная модель остаётся в float32
    assert all(dtype == torch.float32 for dtype in _collect_dtypes(model))


def test_apply_smooth_quant_registers_scales():
    class SmoothNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 4, kernel_size=3)
            self.linear = nn.Linear(36, 2)

        def forward(self, x):
            x = self.conv(x)
            x = torch.relu(x)
            x = torch.flatten(x, 1)
            return self.linear(x)

    model = SmoothNet()
    calib_data = [torch.randn(1, 3, 5, 5) for _ in range(2)]
    quant.apply_smooth_quant(model, calib_data, alpha=0.5, max_batches=2)
    assert hasattr(model.conv, "smooth_quant_scale")
    assert hasattr(model.linear, "smooth_quant_scale")
    assert model.conv.smooth_quant_scale.ndim == 1
    assert model.linear.smooth_quant_scale.ndim == 1


def test_apply_smooth_quant_validates_alpha():
    model = LatencyNet()
    with pytest.raises(ValueError):
        quant.apply_smooth_quant(model, [], alpha=1.5)
