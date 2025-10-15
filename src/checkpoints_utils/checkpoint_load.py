from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, MutableMapping

import torch
from omegaconf import DictConfig, OmegaConf
from Drons.src.models.create_model import create_model  


def _load_checkpoint(path: Path | str, map_location: str | torch.device | None):
    """Load a PyTorch Lightning checkpoint."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint '{path}' does not exist")
    return torch.load(path, map_location=map_location, weights_only=False)


def _ensure_cfg(cfg: Any) -> DictConfig:
    """Convert configs saved in checkpoints to ``DictConfig`` instances."""
    if isinstance(cfg, DictConfig):
        return cfg
    if isinstance(cfg, Mapping):
        return OmegaConf.create(dict(cfg))
    if hasattr(cfg, "__dict__"):
        return OmegaConf.create(dict(vars(cfg)))
    raise TypeError(
        "Unable to convert checkpoint configuration to DictConfig. "
        "Pass a config explicitly via the `cfg` argument."
    )


def _extract_cfg(ckpt: MutableMapping[str, Any]) -> DictConfig | None:
    hparams = ckpt.get("hyper_parameters")
    if isinstance(hparams, Mapping):
        if "cfg" in hparams:
            try:
                return _ensure_cfg(hparams["cfg"])
            except TypeError:
                pass
    return None


def _infer_num_classes(ckpt: MutableMapping[str, Any]) -> int | None:
    hparams = ckpt.get("hyper_parameters")
    if isinstance(hparams, Mapping):
        for key in ("num_class", "num_classes"):
            if key in hparams:
                try:
                    return int(hparams[key])
                except (TypeError, ValueError):
                    continue

    state_dict = ckpt.get("state_dict", {})
    if not isinstance(state_dict, Mapping):
        return None

    candidate: int | None = None
    for tensor in state_dict.values():
        if not isinstance(tensor, torch.Tensor) or tensor.ndim != 2:
            continue
        out_features, in_features = tensor.shape
        if out_features <= 1 or in_features <= 1:
            continue
        if candidate is None or out_features > candidate:
            candidate = int(out_features)
    return candidate


def load_model_from_checkpoint(
    checkpoint_path: Path | str,
    *,
    cfg: Any | None = None,
    num_classes: int | None = None,
    map_location: str | torch.device | None = "cpu",
    strict: bool = True,
) -> torch.nn.Module:
    """Restore a Lightning classification model from a saved checkpoint.

    Parameters
    ----------
    checkpoint_path:
        Path to the ``.ckpt`` file produced by PyTorch Lightning.
    cfg:
        Configuration used to build the model. If omitted the loader will try
        to recover it from the checkpoint ``hyper_parameters`` section.
    num_classes:
        Number of output classes. When ``None`` the loader attempts to infer
        the value from the checkpoint meta-data or classifier head weights.
    map_location:
        Device mapping passed to :func:`torch.load`.
    strict:
        Passed to :meth:`torch.nn.Module.load_state_dict`.

    Returns
    -------
    torch.nn.Module
        The restored Lightning module ready for inference (in ``eval`` mode).
    """

    checkpoint = _load_checkpoint(checkpoint_path, map_location=map_location)

    if cfg is None:
        cfg = _extract_cfg(checkpoint)
        if cfg is None:
            raise ValueError(
                "Configuration was not provided and could not be recovered "
                "from the checkpoint. Please pass `cfg` explicitly."
            )
    else:
        cfg = _ensure_cfg(cfg)

    if num_classes is None:
        num_classes = _infer_num_classes(checkpoint)
        if num_classes is None:
            raise ValueError(
                "Could not infer the number of classes from the checkpoint. "
                "Pass it explicitly via the `num_classes` argument."
            )

    model = create_model(cfg, num_class=num_classes)
    state_dict = checkpoint.get("state_dict")
    if not isinstance(state_dict, Mapping):
        raise KeyError("Checkpoint does not contain a valid 'state_dict'")
    bad_prefixes = ("loss.", "criterion.", "metrics.", "optimizer.", "scheduler.")
    state_dict = {k: v for k, v in state_dict.items() if not k.startswith(bad_prefixes)}

    # (optional) strip DP prefix if present
    if any(k.startswith("module.") for k in state_dict):
        state_dict = {k[len("module."):] if k.startswith("module.") else k: v
                    for k, v in state_dict.items()}

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print("missing checkpoints states:", missing[:10])
    print("unexpected checkpoints state:", unexpected[:10])
    model.load_state_dict(state_dict, strict=strict)
    model.eval()
    if map_location is not None:
        device = torch.device(map_location)
        model.to(device)

    return model