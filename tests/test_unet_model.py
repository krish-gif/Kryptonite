"""Pytest unit tests for 2D SeaIceUNet model architecture."""

import numpy as np
import pytest
import torch

from unet_model import SeaIceUNet


def test_forward_pass_shape():
    """SeaIceUNet output shape must match (Batch, out_channels, H, W)."""
    batch_size = 4
    in_channels = 7
    out_channels = 1
    h, w = 64, 64

    model = SeaIceUNet(in_channels=in_channels, out_channels=out_channels, base_filters=16, depth=3)
    x = torch.randn(batch_size, in_channels, h, w)

    out = model(x)
    assert out.shape == (batch_size, out_channels, h, w)
    assert out.dtype == torch.float32


def test_output_range_is_bounded():
    """Outputs must strictly lie in [0.0, 1.0] due to Sigmoid activation."""
    model = SeaIceUNet(in_channels=5, out_channels=1, base_filters=16, depth=3)
    # Test with extreme values to test saturation
    x = torch.randn(2, 5, 32, 32) * 50.0

    out = model(x)
    assert out.min().item() >= 0.0
    assert out.max().item() <= 1.0
    assert not torch.isnan(out).any()


def test_land_mask_enforcement():
    """Land pixels must be identically 0.0 when land_mask is registered."""
    h, w = 32, 32
    # Create mask where bottom half is land (0) and top half is ocean (1)
    land_mask = np.ones((h, w), dtype=np.float32)
    land_mask[h // 2 :, :] = 0.0
    mask_tensor = torch.from_numpy(land_mask)

    model = SeaIceUNet(in_channels=3, out_channels=1, base_filters=16, depth=3, land_mask=mask_tensor)
    x = torch.randn(2, 3, h, w)

    out = model(x)
    # Check bottom half is exactly 0.0
    land_predictions = out[:, :, h // 2 :, :]
    assert torch.all(land_predictions == 0.0)

    # Check top half has non-zero predictions
    ocean_predictions = out[:, :, : h // 2, :]
    assert torch.any(ocean_predictions > 0.0)


def test_odd_and_non_power_of_two_dimensions():
    """Model must handle arbitrary non-power-of-two and odd grid dimensions without crashing."""
    odd_shapes = [
        (67, 53),  # Both dimensions odd/prime
        (75, 75),  # Odd square
        (45, 62),  # One odd, one even
    ]

    model = SeaIceUNet(in_channels=7, out_channels=1, base_filters=16, depth=4)

    for h, w in odd_shapes:
        x = torch.randn(2, 7, h, w)
        out = model(x)
        assert out.shape == (2, 1, h, w), f"Failed on grid shape ({h}, {w})"


def test_channel_mismatch_raises_error():
    """Model must raise ValueError if input channel count does not match in_channels."""
    model = SeaIceUNet(in_channels=7, out_channels=1, base_filters=16, depth=3)
    wrong_channel_x = torch.randn(2, 5, 32, 32)

    with pytest.raises(ValueError, match="Input channel count mismatch"):
        model(wrong_channel_x)


def test_backward_gradient_flow():
    """Model must allow backward propagation with valid gradients."""
    model = SeaIceUNet(in_channels=4, out_channels=1, base_filters=16, depth=2)
    x = torch.randn(2, 4, 32, 32, requires_grad=True)
    target = torch.rand(2, 1, 32, 32)

    out = model(x)
    loss = torch.nn.functional.mse_loss(out, target)
    loss.backward()

    # Check that gradients exist and are finite
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Gradient missing for {name}"
            assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"


def test_bilinear_upsampling_mode():
    """Model with bilinear=True should produce matching output shape."""
    model = SeaIceUNet(in_channels=3, out_channels=1, base_filters=16, depth=3, bilinear=True)
    x = torch.randn(2, 3, 40, 40)
    out = model(x)
    assert out.shape == (2, 1, 40, 40)


def test_parameter_count_default_config():
    """Default config (base_filters=32, depth=4, in_channels=7) should have ~7.76M params."""
    model = SeaIceUNet(in_channels=7, out_channels=1, base_filters=32, depth=4)
    param_count = model.count_parameters()
    # Expected exact: 7,763,713
    assert 7_500_000 < param_count < 8_000_000, f"Unexpected parameter count: {param_count}"
