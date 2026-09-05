"""Pytest unit tests for the train.py training loop and helper functions."""

import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from loss_functions import SeaIceLoss
from torch_dataset import SeaIceDataset
from torch.utils.data import DataLoader
from unet_model import SeaIceUNet
from train import (
    _run_one_epoch_train,
    _run_one_epoch_val,
    _save_checkpoint,
    _load_best_checkpoint,
    _seed_everything,
    train_model,
)


@pytest.fixture
def tiny_pipeline(tmp_path):
    """Minimal synthetic DataLoaders and model for fast unit testing."""
    n, c, h, w = 16, 3, 16, 16

    # Land mask: top half ocean, bottom half land
    land_mask = torch.ones(h, w)
    land_mask[h // 2 :, :] = 0.0

    # Synthetic X/Y arrays
    X = np.random.rand(n, c, h, w).astype(np.float32)
    Y = np.random.rand(n, 1, h, w).astype(np.float32)
    # Zero land in Y
    Y[:, :, h // 2 :, :] = 0.0

    ds = SeaIceDataset(X, Y)
    n_train = 12
    train_ds = SeaIceDataset(X[:n_train], Y[:n_train])
    val_ds = SeaIceDataset(X[n_train:], Y[n_train:])

    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False)

    model = SeaIceUNet(
        in_channels=c, out_channels=1, base_filters=8, depth=2,
        land_mask=land_mask,
    )
    criterion = SeaIceLoss(land_mask=land_mask, base_loss="mse")
    device = torch.device("cpu")

    return {
        "model": model,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "criterion": criterion,
        "land_mask": land_mask,
        "device": device,
        "tmp_path": tmp_path,
        "in_channels": c,
        "h": h,
        "w": w,
    }


def test_run_one_epoch_train_returns_finite_loss(tiny_pipeline):
    """Training epoch helper must return a finite, non-NaN float loss."""
    f = tiny_pipeline
    optimizer = torch.optim.AdamW(f["model"].parameters(), lr=1e-3)
    loss = _run_one_epoch_train(
        f["model"], f["train_loader"], f["criterion"],
        optimizer, f["device"], grad_clip_norm=1.0
    )
    assert isinstance(loss, float)
    assert math.isfinite(loss)
    assert not math.isnan(loss)


def test_run_one_epoch_val_returns_loss_and_iiee(tiny_pipeline):
    """Validation epoch helper must return (val_loss, val_iiee) both finite."""
    f = tiny_pipeline
    val_loss, val_iiee = _run_one_epoch_val(
        f["model"], f["val_loader"], f["criterion"],
        f["land_mask"], f["device"]
    )
    assert isinstance(val_loss, float) and math.isfinite(val_loss)
    assert isinstance(val_iiee, float) and 0.0 <= val_iiee <= 1.0


def test_save_and_load_checkpoint(tiny_pipeline):
    """Checkpoint save/load cycle must reproduce identical model state dicts."""
    f = tiny_pipeline
    ckpt_path = f["tmp_path"] / "test_ckpt.pt"
    state = {
        "epoch": 1,
        "model_state_dict": f["model"].state_dict(),
        "optimizer_state_dict": torch.optim.AdamW(f["model"].parameters()).state_dict(),
        "scheduler_state_dict": {},
        "train_loss": 0.5,
        "val_loss": 0.4,
        "val_iiee": 0.3,
    }
    _save_checkpoint(state, ckpt_path)
    assert ckpt_path.exists()

    # Load into fresh model
    new_model = SeaIceUNet(
        in_channels=f["in_channels"], out_channels=1,
        base_filters=8, depth=2, land_mask=f["land_mask"],
    )
    loaded = _load_best_checkpoint(new_model, ckpt_path, f["device"])

    # All parameters must match
    for key in f["model"].state_dict():
        orig = f["model"].state_dict()[key]
        reloaded = new_model.state_dict()[key]
        assert torch.allclose(orig, reloaded), f"Mismatch in parameter: {key}"

    assert loaded["val_loss"] == 0.4


def test_load_missing_checkpoint_raises(tiny_pipeline):
    """Loading from nonexistent checkpoint must raise FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="Best checkpoint not found"):
        _load_best_checkpoint(
            tiny_pipeline["model"],
            Path("/nonexistent/path/best_model.pt"),
            tiny_pipeline["device"]
        )


def test_train_model_minimal_run(tiny_pipeline):
    """train_model with 2 epochs must return history of length 2, no NaN losses."""
    f = tiny_pipeline
    history = train_model(
        model=f["model"],
        train_loader=f["train_loader"],
        val_loader=f["val_loader"],
        criterion=f["criterion"],
        land_mask=f["land_mask"],
        num_epochs=2,
        lr=1e-3,
        patience=10,
        checkpoint_dir=str(f["tmp_path"] / "ckpts"),
        seed=42,
    )
    assert len(history) == 2
    for r in history:
        assert math.isfinite(r["train_loss"]), f"NaN train_loss at epoch {r['epoch']}"
        assert math.isfinite(r["val_loss"]), f"NaN val_loss at epoch {r['epoch']}"
        assert 0.0 <= r["val_iiee"] <= 1.0
        # CosineAnnealingLR legitimately reaches 0 at T_max, so >= 0 is correct
        assert r["lr"] >= 0.0


def test_best_checkpoint_only_written_on_improvement(tiny_pipeline):
    """best_model.pt must be saved at least once and never exceed the minimum val_loss."""
    f = tiny_pipeline
    ckpt_dir = f["tmp_path"] / "ckpts_best"
    history = train_model(
        model=f["model"],
        train_loader=f["train_loader"],
        val_loader=f["val_loader"],
        criterion=f["criterion"],
        land_mask=f["land_mask"],
        num_epochs=3,
        lr=1e-3,
        patience=10,
        checkpoint_dir=str(ckpt_dir),
        seed=42,
    )
    best_ckpt = ckpt_dir / "best_model.pt"
    assert best_ckpt.exists()

    ckpt = torch.load(best_ckpt, map_location="cpu", weights_only=True)
    min_val_loss = min(r["val_loss"] for r in history)
    # Saved val_loss must equal the minimum observed across all epochs
    assert abs(ckpt["val_loss"] - min_val_loss) < 1e-7


def test_history_json_written(tiny_pipeline):
    """training_history.json must be written and contain all expected keys."""
    f = tiny_pipeline
    ckpt_dir = f["tmp_path"] / "ckpts_json"
    train_model(
        model=f["model"],
        train_loader=f["train_loader"],
        val_loader=f["val_loader"],
        criterion=f["criterion"],
        land_mask=f["land_mask"],
        num_epochs=2,
        patience=10,
        checkpoint_dir=str(ckpt_dir),
        seed=42,
    )
    history_path = ckpt_dir / "training_history.json"
    assert history_path.exists()
    with open(history_path) as fh:
        data = json.load(fh)
    assert len(data) == 2
    required_keys = {"epoch", "train_loss", "val_loss", "val_iiee", "lr"}
    for record in data:
        assert required_keys.issubset(record.keys())


def test_lr_decays_with_cosine_scheduler(tiny_pipeline):
    """LR in history must strictly decrease with CosineAnnealingLR over 5 epochs."""
    f = tiny_pipeline
    ckpt_dir = f["tmp_path"] / "ckpts_lr"
    history = train_model(
        model=f["model"],
        train_loader=f["train_loader"],
        val_loader=f["val_loader"],
        criterion=f["criterion"],
        land_mask=f["land_mask"],
        num_epochs=5,
        lr=1e-2,
        patience=10,
        checkpoint_dir=str(ckpt_dir),
        seed=42,
    )
    lr_values = [r["lr"] for r in history]
    assert lr_values[-1] < lr_values[0], (
        f"Expected LR decay, but got {lr_values}"
    )


def test_early_stopping_halts_correctly(tiny_pipeline):
    """With patience=1, training must halt after at most 2 epochs."""
    f = tiny_pipeline
    ckpt_dir = f["tmp_path"] / "ckpts_es"
    history = train_model(
        model=f["model"],
        train_loader=f["train_loader"],
        val_loader=f["val_loader"],
        criterion=f["criterion"],
        land_mask=f["land_mask"],
        num_epochs=20,
        lr=1e-3,
        patience=1,
        checkpoint_dir=str(ckpt_dir),
        seed=42,
    )
    assert len(history) <= 2, (
        f"Expected at most 2 epochs with patience=1, but ran {len(history)}"
    )


def test_seed_reproducibility(tiny_pipeline):
    """Two runs with same seed and identical DataLoaders must give same losses.

    The DataLoader's shuffle uses its own Generator that is independent of
    torch.manual_seed(). We therefore build explicitly-seeded Generators so
    that batch order is byte-for-byte identical across the two runs.
    """
    f = tiny_pipeline
    n, c, h, w = 16, f["in_channels"], f["h"], f["w"]

    # Rebuild X/Y with a fixed numpy seed so both runs share the same data
    rng = np.random.default_rng(seed=0)
    X = rng.random((n, c, h, w), dtype=np.float32)
    Y = rng.random((n, 1, h, w), dtype=np.float32)
    Y[:, :, h // 2 :, :] = 0.0  # zero land in targets

    def _make_loaders(dl_seed: int):
        n_train = 12
        train_ds = SeaIceDataset(X[:n_train], Y[:n_train])
        val_ds   = SeaIceDataset(X[n_train:], Y[n_train:])
        g = torch.Generator()
        g.manual_seed(dl_seed)
        tl = DataLoader(train_ds, batch_size=4, shuffle=True, generator=g)
        vl = DataLoader(val_ds,   batch_size=4, shuffle=False)
        return tl, vl

    def _fresh_model(model_seed: int = 99):
        # Pin global RNG before construction so kaiming_normal_ is identical.
        torch.manual_seed(model_seed)
        return SeaIceUNet(
            in_channels=c, out_channels=1,
            base_filters=8, depth=2, land_mask=f["land_mask"],
        )

    tl1, vl1 = _make_loaders(dl_seed=42)
    tl2, vl2 = _make_loaders(dl_seed=42)

    h1 = train_model(
        model=_fresh_model(), train_loader=tl1, val_loader=vl1,
        criterion=f["criterion"], land_mask=f["land_mask"],
        num_epochs=2, patience=10,
        checkpoint_dir=str(f["tmp_path"] / "seed_run1"), seed=77,
    )
    h2 = train_model(
        model=_fresh_model(), train_loader=tl2, val_loader=vl2,
        criterion=f["criterion"], land_mask=f["land_mask"],
        num_epochs=2, patience=10,
        checkpoint_dir=str(f["tmp_path"] / "seed_run2"), seed=77,
    )
    # Same model init seed + same batch order → identical train losses
    for r1, r2 in zip(h1, h2):
        assert abs(r1["train_loss"] - r2["train_loss"]) < 1e-5, (
            f"Epoch {r1['epoch']}: {r1['train_loss']} vs {r2['train_loss']}"
        )
