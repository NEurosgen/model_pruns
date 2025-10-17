import pytest
import torch
import torch.nn as nn

from src.pruning import channel_pruning as cp


class TinyConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 4, kernel_size=3)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(4, 4, kernel_size=3)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        return self.conv2(x)


class DepthwiseNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.dw = nn.Conv2d(4, 4, kernel_size=3, groups=4)
        self.pw = nn.Conv2d(4, 8, kernel_size=1)

    def forward(self, x):
        return self.pw(self.dw(x))


def test_collect_ignored_layers_includes_depthwise_and_patterns():
    model = DepthwiseNet()
    ignored = cp.collect_ignored_layers(model, ["pw"])
    assert model.dw in ignored
    assert model.pw in ignored


def test_normalize_example_inputs_variants():
    tensor = torch.randn(1, 3, 4, 4)
    moved = cp._normalize_example_inputs(tensor, "cpu")
    assert torch.equal(tensor, moved)

    seq = [tensor.clone(), tensor.clone()]
    moved_seq = cp._normalize_example_inputs(seq, "cpu")
    assert all(t.device.type == "cpu" for t in moved_seq)

    mapping = {"x": tensor.clone()}
    moved_map = cp._normalize_example_inputs(mapping, "cpu")
    assert torch.equal(moved_map["x"], tensor)

    with pytest.raises(TypeError):
        cp._normalize_example_inputs(object(), "cpu")


def test_resolve_importance_accepts_strings():
    importance = cp._resolve_importance("l2")
    assert getattr(importance, "p", None) == 2

    with pytest.raises(ValueError):
        cp._resolve_importance("unknown")

    class CustomImportance(cp.tp.importance.Importance):
        pass

    inst = CustomImportance()
    assert cp._resolve_importance(inst) is inst


def _count_zero_filters(module: nn.Conv2d) -> int:
    weight = module.weight.detach()
    return int((weight.view(weight.shape[0], -1).abs().sum(dim=1) == 0).sum().item())


def test_prune_model_channels_structured():
    model = TinyConvNet()
    example = torch.randn(1, 3, 10, 10)
    cp.prune_model_channels(
        model,
        example,
        amount=0.5,
        exclude_name_patterns=[],
        min_out_channels=1,
        verbose=False,
    )
    assert _count_zero_filters(model.conv1) >= 2


def test_prune_model_channels_global():
    model = TinyConvNet()
    example = torch.randn(1, 3, 10, 10)
    cp.prune_model_channels(
        model,
        example,
        amount=0.25,
        exclude_name_patterns=[],
        min_out_channels=1,
        global_pruning=True,
        verbose=False,
    )
    assert _count_zero_filters(model.conv1) >= 1
    assert _count_zero_filters(model.conv2) >= 1


def test_prune_model_channels_amount_validation():
    model = TinyConvNet()
    example = torch.randn(1, 3, 10, 10)
    with pytest.raises(ValueError):
        cp.prune_model_channels(model, example, amount=1.5)


def test_progressive_channel_pruning_runs_multiple_steps():
    model = TinyConvNet()
    example = torch.randn(1, 3, 10, 10)
    cp.progressive_channel_pruning(
        model,
        example,
        schedule=[0.25, 0.5],
        exclude_name_patterns=[],
        min_out_channels=1,
        verbose=False,
    )
    assert _count_zero_filters(model.conv2) >= 2


def test_progressive_channel_pruning_validates_schedule():
    model = TinyConvNet()
    example = torch.randn(1, 3, 10, 10)
    with pytest.raises(ValueError):
        cp.progressive_channel_pruning(model, example, schedule=[], verbose=False)
    with pytest.raises(ValueError):
        cp.progressive_channel_pruning(model, example, schedule=[0.2], warmup_steps=-1, verbose=False)
    with pytest.raises(ValueError):
        cp.progressive_channel_pruning(model, example, schedule=[0.0, 1.0], verbose=False)
