"""Pytest unit tests for SeaIceLoss and compute_iiee metric."""

import numpy as np
import pytest
import torch

from loss_functions import SeaIceLoss, compute_iiee


@pytest.fixture
def synthetic_tensors():
    """Create deterministic synthetic prediction, target, and land mask."""
    b, c, h, w = 4, 1, 32, 32

    # Top half ocean (1), bottom half land (0)
    land_mask = torch.ones((h, w), dtype=torch.float32)
    land_mask[h // 2 :, :] = 0.0

    # Target: linear gradient in ocean from 0.0 to 1.0
    target = torch.zeros((b, c, h, w), dtype=torch.float32)
    for y in range(h // 2):
        target[:, :, y, :] = float(y) / float(h // 2)

    # Pred: target plus small offset
    pred = target.clone() + 0.05
    pred.requires_grad = True

    return {
        "land_mask": land_mask,
        "target": target,
        "pred": pred,
        "b": b,
        "h": h,
        "w": w,
    }


def test_differentiable_graph_and_scalar(synthetic_tensors):
    """SeaIceLoss must return a 0-dim scalar tensor with valid grad_fn."""
    criterion = SeaIceLoss(land_mask=synthetic_tensors["land_mask"], base_loss="mse")
    loss = criterion(synthetic_tensors["pred"], synthetic_tensors["target"])

    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0, f"Expected scalar loss, got shape {loss.shape}"
    assert loss.requires_grad is True, "Loss must require gradients!"
    assert loss.grad_fn is not None, "Loss must have a valid autograd grad_fn!"


def test_perfect_prediction_zero_loss(synthetic_tensors):
    """L(target, target) must evaluate to exactly 0.0 for both MSE and MAE."""
    target = synthetic_tensors["target"]
    mask = synthetic_tensors["land_mask"]

    criterion_mse = SeaIceLoss(land_mask=mask, base_loss="mse")
    criterion_mae = SeaIceLoss(land_mask=mask, base_loss="mae")

    loss_mse = criterion_mse(target, target)
    loss_mae = criterion_mae(target, target)

    assert loss_mse.item() == 0.0
    assert loss_mae.item() == 0.0


def test_boundary_weight_map_statistics(synthetic_tensors):
    """Weight map must have min=1.0 on ocean, max=edge_weight on edge, 0.0 on land."""
    target = synthetic_tensors["target"]
    mask = synthetic_tensors["land_mask"]
    h = synthetic_tensors["h"]

    criterion = SeaIceLoss(
        land_mask=mask,
        edge_weight=3.5,
        edge_band=0.05,
        edge_threshold=0.15,
    )
    weights = criterion.get_boundary_weight_map(target)

    # Detached check
    assert weights.requires_grad is False

    # Land pixels must be 0.0
    land_weights = weights[:, :, h // 2 :, :]
    assert torch.all(land_weights == 0.0)

    # Ocean pixels must be >= 1.0 and <= 3.5
    ocean_weights = weights[:, :, : h // 2, :]
    assert ocean_weights.min().item() == 1.0
    assert ocean_weights.max().item() == 3.5

    # Confirm edge pixels actually exist
    edge_count = (ocean_weights == 3.5).sum().item()
    assert edge_count > 0, "No pixels flagged as edge in target gradient!"


def test_compute_iiee_non_differentiable(synthetic_tensors):
    """compute_iiee must return a float in [0.0, 1.0] without autograd tracking."""
    pred = synthetic_tensors["pred"]
    target = synthetic_tensors["target"]
    mask = synthetic_tensors["land_mask"]

    iiee = compute_iiee(pred, target, land_mask=mask, threshold=0.15)
    assert isinstance(iiee, float)
    assert not hasattr(iiee, "grad_fn")
    assert 0.0 <= iiee <= 1.0


def test_compute_iiee_perfect_match(synthetic_tensors):
    """IIEE on identical inputs must be exactly 0.0."""
    target = synthetic_tensors["target"]
    mask = synthetic_tensors["land_mask"]

    iiee = compute_iiee(target, target, land_mask=mask, threshold=0.15)
    assert iiee == 0.0


def test_land_pixel_invariance(synthetic_tensors):
    """Overwriting predictions over land with garbage values must not alter loss or IIEE."""
    pred = synthetic_tensors["pred"]
    target = synthetic_tensors["target"]
    mask = synthetic_tensors["land_mask"]
    h = synthetic_tensors["h"]

    criterion = SeaIceLoss(land_mask=mask, base_loss="mse")

    loss_clean = criterion(pred, target).item()
    iiee_clean = compute_iiee(pred, target, land_mask=mask)

    # Corrupt land coordinates (bottom half) with extreme numbers
    corrupted_pred = pred.clone()
    corrupted_pred[:, :, h // 2 :, :] = 888888.0

    loss_corrupted = criterion(corrupted_pred, target).item()
    iiee_corrupted = compute_iiee(corrupted_pred, target, land_mask=mask)

    assert abs(loss_clean - loss_corrupted) < 1e-6
    assert abs(iiee_clean - iiee_corrupted) < 1e-6


def test_degenerate_land_mask_raises():
    """All-land mask must raise ValueError."""
    all_land_mask = torch.zeros((16, 16), dtype=torch.float32)

    with pytest.raises(ValueError, match="contains 0 ocean pixels"):
        SeaIceLoss(land_mask=all_land_mask)

    with pytest.raises(ValueError, match="contains zero ocean pixels"):
        dummy = torch.ones((1, 1, 16, 16))
        compute_iiee(dummy, dummy, land_mask=all_land_mask)


def test_invalid_parameters_raise():
    """Invalid constructor parameters must raise ValueError."""
    mask = torch.ones((16, 16))

    with pytest.raises(ValueError, match="Unsupported base_loss"):
        SeaIceLoss(land_mask=mask, base_loss="huber")

    with pytest.raises(ValueError, match="edge_weight must be >= 1.0"):
        SeaIceLoss(land_mask=mask, edge_weight=0.5)

    with pytest.raises(ValueError, match="edge_band must be strictly positive"):
        SeaIceLoss(land_mask=mask, edge_band=0.0)


def test_gradient_backprop(synthetic_tensors):
    """Gradients must backpropagate cleanly through pred with finite values."""
    pred = synthetic_tensors["pred"]
    target = synthetic_tensors["target"]
    mask = synthetic_tensors["land_mask"]

    criterion = SeaIceLoss(land_mask=mask, base_loss="mse")
    loss = criterion(pred, target)
    loss.backward()

    assert pred.grad is not None
    assert not torch.isnan(pred.grad).any()
    # Land gradients should be identically zero
    h = synthetic_tensors["h"]
    assert torch.all(pred.grad[:, :, h // 2 :, :] == 0.0)
