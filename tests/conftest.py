import sys
import types

import pytest

torch = pytest.importorskip("torch")
nn = torch.nn


class _DummyImportanceBase:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _DummyMagnitudeImportance(_DummyImportanceBase):
    def __init__(self, p=1):
        super().__init__(p=p)
        self.p = p


class _DummyTaylorImportance(_DummyImportanceBase):
    pass


class _DummyFPGMImportance(_DummyImportanceBase):
    pass


class _DummyMagnitudePruner:
    def __init__(
        self,
        model,
        *,
        example_inputs,
        importance,
        global_pruning,
        ch_sparsity,
        ch_sparsity_dict,
        ignored_layers,
        root_module_types,
    ):
        self.model = model
        self.example_inputs = example_inputs
        self.importance = importance
        self.global_pruning = global_pruning
        self.ch_sparsity = ch_sparsity
        self.ch_sparsity_dict = ch_sparsity_dict or {}
        self.ignored = set(ignored_layers or [])
        self.root_module_types = tuple(root_module_types or ())

    def step(self):
        if self.ch_sparsity_dict:
            items = self.ch_sparsity_dict.items()
        else:
            amount = float(self.ch_sparsity or 0.0)
            items = []
            if amount <= 0.0:
                return
            for module in self.model.modules():
                if isinstance(module, self.root_module_types) and module not in self.ignored:
                    items.append((module, amount))
        for module, amount in items:
            if not hasattr(module, "weight"):
                continue
            weight = module.weight.data
            out_channels = weight.shape[0]
            to_zero = max(1, int(out_channels * float(amount)))
            weight[-to_zero:] = 0
            if getattr(module, "bias", None) is not None:
                module.bias.data[-to_zero:] = 0


if "torch_pruning" not in sys.modules:
    tp_module = types.ModuleType("torch_pruning")
    importance_module = types.SimpleNamespace(
        Importance=_DummyImportanceBase,
        MagnitudeImportance=_DummyMagnitudeImportance,
        TaylorImportance=_DummyTaylorImportance,
        FPGMImportance=_DummyFPGMImportance,
    )
    pruner_module = types.SimpleNamespace(MagnitudePruner=_DummyMagnitudePruner)
    tp_module.importance = importance_module
    tp_module.pruner = pruner_module
    sys.modules["torch_pruning"] = tp_module
