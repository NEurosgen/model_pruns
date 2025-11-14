import math

import pytest
import torch
import torch.nn as nn

from  Optimization.src.pruning import tensor_prunning as tp


class LinearToy(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 4)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(4, 2)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data = torch.arange(16, dtype=torch.float32).view(4, 4)
        self.fc1.bias.data = torch.arange(4, dtype=torch.float32)
        self.fc2.weight.data = torch.arange(8, dtype=torch.float32).view(2, 4) + 100
        self.fc2.bias.data = torch.arange(2, dtype=torch.float32)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class ConvToy(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, kernel_size=3)
        self.reset_parameters()

    def reset_parameters(self):
        weight = torch.arange(3 * 4 * 3 * 3, dtype=torch.float32).view(4, 3, 3, 3)
        self.conv.weight.data = weight
        if self.conv.bias is not None:
            self.conv.bias.data.zero_()

    def forward(self, x):
        return self.conv(x)


@pytest.fixture
def linear_model():
    return LinearToy()


@pytest.fixture
def conv_model():
    return ConvToy()


def count_weight_zeros(model: nn.Module) -> int:
    zeros = 0
    for module in model.modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            zeros += torch.count_nonzero(module.weight == 0).item()
    return zeros


def test_global_unstructured_prune(linear_model):
    total_weights = sum(module.weight.numel() for module in linear_model.modules() if isinstance(module, nn.Linear))
    tp.global_unstructured_prune(linear_model, amount=0.25)
    zeros = count_weight_zeros(linear_model)
    assert zeros == math.floor(total_weights * 0.25)


def test_global_unstructured_prune_no_targets():
    class Identity(nn.Module):
        def forward(self, x):
            return x

    model = nn.Sequential(Identity())
    with pytest.raises(RuntimeError):
        tp.global_unstructured_prune(model, amount=0.2)


def test_layer_unstructured_prune_sets_mask_per_module(linear_model):
    tp.layer_unstructured_prune(linear_model, amount=0.5)
    for module in linear_model.modules():
        if isinstance(module, nn.Linear):
            numel = module.weight.numel()
            zeros = torch.count_nonzero(module.weight == 0).item()
            assert zeros == math.floor(numel * 0.5)


def test_structured_filter_prune_removes_complete_filters(conv_model):
    tp.structured_filter_prune(conv_model, amount=0.5)
    weight = conv_model.conv.weight.detach()
    channel_norms = weight.view(weight.shape[0], -1).abs().sum(dim=1)
    pruned = (channel_norms == 0).nonzero().flatten().tolist()
    assert len(pruned) == 2


def test_random_unstructured_prune_deterministic(linear_model):
    torch.manual_seed(0)
    tp.random_unstructured_prune(linear_model, amount=0.25)
    zeros_first = count_weight_zeros(linear_model)
    linear_model.reset_parameters()
    torch.manual_seed(0)
    tp.random_unstructured_prune(linear_model, amount=0.25)
    assert count_weight_zeros(linear_model) == zeros_first


def test_threshold_prune_affects_bias(linear_model):
    linear_model.fc1.weight.data.fill_(0.1)
    linear_model.fc1.bias.data.fill_(0.05)
    tp.threshold_prune(linear_model, threshold=0.09, prune_bias=True)
    assert torch.count_nonzero(linear_model.fc1.weight).item() == 0
    assert torch.count_nonzero(linear_model.fc1.bias).item() == 0


def test_report_sparsity_aggregate(linear_model):
    tp.layer_unstructured_prune(linear_model, amount=0.5)
    stats = tp.report_sparsity(linear_model)
    assert "global" in stats
    assert 0 <= stats["global"] <= 1
    assert any(name.endswith("fc1") for name in stats)


def test_make_pruning_permanent_removes_reparam(linear_model):
    tp.layer_unstructured_prune(linear_model, amount=0.25)
    tp.make_pruning_permanent(linear_model)
    for module in linear_model.modules():
        if isinstance(module, nn.Linear):
            assert not hasattr(module, "weight_mask")
            assert not hasattr(module, "weight_orig")
