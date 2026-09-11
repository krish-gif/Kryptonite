"""2D U-Net Model for Sea-Ice Concentration Forecasting.

Implements a resolution-agnostic 2D U-Net (`SeaIceUNet`) that predicts the next day's
sea-ice concentration map from a channel-stacked temporal window of past daily maps.
Features:
- Arbitrary non-power-of-2 grid handling via center padding before skip concatenation.
- Configurable base filters and depth.
- Kaiming/He normal weight initialization.
- Optional static land-mask buffer registration for physical constraint enforcement.
"""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


logger = logging.getLogger("SeaIceUNet")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _formatter = logging.Formatter(
        "[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    _handler.setFormatter(_formatter)
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)


class DoubleConv(nn.Module):
    """Building block: [Conv2d(3x3, pad=1) -> BatchNorm2d -> ReLU] x 2."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mid_channels: int | None = None,
    ) -> None:
        super().__init__()
        mid = mid_channels or out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling block: MaxPool2d(2) followed by DoubleConv."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (downsampled_feature, skip_feature)."""
        skip = self.conv(x)
        down = self.pool(skip)
        return down, skip


class Up(nn.Module):
    """Upscaling block: ConvTranspose2d (or Bilinear Upsample) + Pad + Concat + DoubleConv.

    Handles non-power-of-2 and odd spatial dimensions by center-padding the upsampled
    feature map to match the exact spatial dimensions of the skip connection.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bilinear: bool = False,
    ) -> None:
        super().__init__()
        self.bilinear = bilinear

        if bilinear:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                nn.Conv2d(in_channels, in_channels // 2, kernel_size=1, bias=False),
            )
            self.conv = DoubleConv(in_channels, out_channels)
        else:
            # in_channels is the channels of the deeper feature map.
            # Upsample halves channels: in_channels -> in_channels // 2
            # After concatenating with skip connection (which has in_channels // 2 channels),
            # total channels fed to DoubleConv is in_channels.
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2
            )
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)

        # Handle non-power-of-2 / odd grid dimensions:
        # If the input grid dimension was odd (e.g. 65), pooling gave floor(65/2) = 32.
        # Upsampling 32 by stride 2 gives 64, which is 1 pixel smaller than the skip connection (65).
        # We pad the upsampled tensor x symmetrically to match skip's exact (Height, Width).
        diff_y = skip.size()[2] - x.size()[2]
        diff_x = skip.size()[3] - x.size()[3]

        if diff_x != 0 or diff_y != 0:
            # F.pad format: [pad_left, pad_right, pad_top, pad_bottom]
            x = F.pad(
                x,
                [
                    diff_x // 2,
                    diff_x - diff_x // 2,
                    diff_y // 2,
                    diff_y - diff_y // 2,
                ],
            )

        # Concatenate along channel dimension
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """1x1 convolution mapping to out_channels followed by Sigmoid activation."""

    def __init__(self, in_channels: int, out_channels: int = 1) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sigmoid(self.conv(x))


class SeaIceUNet(nn.Module):
    """Resolution-agnostic 2D U-Net for sea-ice concentration forecasting.

    Accepts temporal window input: (Batch, in_channels, Height, Width)
    where in_channels corresponds to the sliding-window size (e.g. 7 days).
    Outputs next-day forecast: (Batch, out_channels, Height, Width) in [0.0, 1.0].

    Features:
    - Arbitrary spatial dimension compatibility (supports odd / non-power-of-2 grids).
    - Configurable depth and base filter multiplier.
    - Kaiming/He normal weight initialization.
    - Optional static land mask buffer for enforcing physical zero-ice constraints on land.

    Args:
        in_channels: Number of past observation maps stacked as channels (e.g. 7).
        out_channels: Number of target prediction channels (default: 1).
        base_filters: Number of filters in first encoder level (default: 32).
        depth: Number of downsampling / upsampling stages (default: 4).
        land_mask: Optional static land mask tensor (Height, Width) where ocean=1, land=0.
        bilinear: If True, uses bilinear interpolation for upsampling instead of
                  learnable ConvTranspose2d (default: False).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 1,
        base_filters: int = 32,
        depth: int = 4,
        land_mask: torch.Tensor | None = None,
        bilinear: bool = False,
    ) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError(f"depth must be >= 1, got {depth}")
        if base_filters < 1:
            raise ValueError(f"base_filters must be >= 1, got {base_filters}")
        if in_channels < 1:
            raise ValueError(f"in_channels must be >= 1, got {in_channels}")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_filters = base_filters
        self.depth = depth
        self.bilinear = bilinear

        # 1. Register land mask as non-trainable buffer
        if land_mask is not None:
            if not isinstance(land_mask, torch.Tensor):
                land_mask = torch.tensor(land_mask, dtype=torch.float32)
            else:
                land_mask = land_mask.to(dtype=torch.float32)

            # Ensure mask is shaped (1, 1, H, W) for direct broadcasting
            if land_mask.ndim == 2:
                land_mask = land_mask.unsqueeze(0).unsqueeze(0)
            elif land_mask.ndim == 3:
                land_mask = land_mask.unsqueeze(0)

            # Enforce binary ocean (1) vs land (0)
            land_mask = (land_mask != 0.0).float()
            self.register_buffer("land_mask", land_mask)
        else:
            self.register_buffer("land_mask", None)

        # 2. Build Encoder Path
        self.inc = DoubleConv(in_channels, base_filters)
        self.down_blocks = nn.ModuleList()
        curr_filters = base_filters

        for i in range(depth - 1):
            next_filters = curr_filters * 2
            self.down_blocks.append(
                nn.Sequential(
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    DoubleConv(curr_filters, next_filters),
                )
            )
            curr_filters = next_filters

        # 3. Bottleneck at deepest point
        bottleneck_in = curr_filters
        bottleneck_out = curr_filters * 2
        self.bottleneck_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bottleneck = DoubleConv(bottleneck_in, bottleneck_out)

        # 4. Build Decoder Path
        self.up_blocks = nn.ModuleList()
        dec_in = bottleneck_out
        for i in range(depth):
            dec_out = dec_in // 2
            self.up_blocks.append(Up(dec_in, dec_out, bilinear=bilinear))
            dec_in = dec_out

        # 5. Output Head
        self.outc = OutConv(base_filters, out_channels)

        # 6. Apply Kaiming/He Normal Weight Initialization
        self._init_weights()

        total_params = self.count_parameters()
        logger.info(
            f"SeaIceUNet initialized: in_channels={in_channels}, out_channels={out_channels}, "
            f"base_filters={base_filters}, depth={depth}, bilinear={bilinear}, "
            f"has_land_mask={self.land_mask is not None} -> Total Parameters: {total_params:,}"
        )

    def _init_weights(self) -> None:
        """Initialize all conv and batchnorm weights using Kaiming/He normal scheme."""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def count_parameters(self) -> int:
        """Return total number of trainable parameters in the model."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through U-Net with skip connections and land masking.

        Args:
            x: Input tensor of shape (Batch, in_channels, Height, Width).

        Returns:
            Output prediction of shape (Batch, out_channels, Height, Width) in [0.0, 1.0].
        """
        if x.ndim != 4:
            raise ValueError(
                f"Expected 4D input tensor (Batch, Channels, Height, Width), but got shape {x.shape}"
            )
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"Input channel count mismatch: model expected {self.in_channels} channels, "
                f"but received input with {x.shape[1]} channels."
            )

        # 1. Encoder Path
        skips: list[torch.Tensor] = []
        x1 = self.inc(x)
        skips.append(x1)

        curr = x1
        for down in self.down_blocks:
            curr = down(curr)
            skips.append(curr)

        # 2. Bottleneck
        b_down = self.bottleneck_pool(curr)
        b_feat = self.bottleneck(b_down)

        # 3. Decoder Path with Skip Connections (in reverse order)
        curr = b_feat
        for up_block in self.up_blocks:
            skip_feat = skips.pop()
            curr = up_block(curr, skip_feat)

        # 4. Output Head (1x1 Conv + Sigmoid)
        logits_sigmoid = self.outc(curr)

        # 5. Apply Static Land Mask (if provided)
        if self.land_mask is not None:
            # Broadcast mask: (1, 1, H, W) * (Batch, 1, H, W)
            # If input spatial dimensions differ from mask, interpolate mask or raise
            if self.land_mask.shape[-2:] != logits_sigmoid.shape[-2:]:
                mask = F.interpolate(
                    self.land_mask,
                    size=logits_sigmoid.shape[-2:],
                    mode="nearest",
                )
            else:
                mask = self.land_mask
            output = logits_sigmoid * mask
        else:
            output = logits_sigmoid

        return output

    def forward_checkpointed(self, x: torch.Tensor) -> torch.Tensor:
        """Memory-efficient forward pass using gradient checkpointing.

        Applies ``torch.utils.checkpoint.checkpoint`` to the bottleneck and
        every decoder up-block — the activation-heavy parts of the U-Net.  The
        encoder path (inc + down_blocks) is left uncheckpointed because its
        skip-connection tensors must remain materialised for the decoder to
        read from.

        This method is called exclusively by ``local_train.py`` when
        ``LocalTrainConfig.use_grad_checkpoint = True``.  It is **never**
        called by the existing ``train.py`` / Lightning AI path — those
        continue to call ``forward()`` as before.

        Gradient checkpointing trades additional forward re-computation during
        the backward pass for a significant reduction in peak VRAM usage, which
        is critical for fitting the model on a 6 GB laptop GPU.

        Args:
            x: Input tensor of shape (Batch, in_channels, Height, Width).

        Returns:
            Output prediction of shape (Batch, out_channels, Height, Width)
            in [0.0, 1.0], identical to ``forward()``.
        """
        import torch.utils.checkpoint as torch_ckpt

        if x.ndim != 4:
            raise ValueError(
                f"Expected 4D input tensor (Batch, Channels, Height, Width), "
                f"but got shape {x.shape}"
            )
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"Input channel count mismatch: model expected {self.in_channels} "
                f"channels, but received input with {x.shape[1]} channels."
            )

        # ── Encoder (not checkpointed — skips must stay alive for decoder) ──
        skips: list[torch.Tensor] = []
        x1 = self.inc(x)
        skips.append(x1)

        curr = x1
        for down in self.down_blocks:
            curr = down(curr)
            skips.append(curr)

        # ── Bottleneck (checkpointed) ────────────────────────────────────────
        # checkpoint() requires inputs that require grad; pool output may not
        # have requires_grad=True in eval mode, so we gate on torch.is_grad_enabled.
        b_down = self.bottleneck_pool(curr)

        def _run_bottleneck(inp: torch.Tensor) -> torch.Tensor:
            return self.bottleneck(inp)

        b_feat = torch_ckpt.checkpoint(_run_bottleneck, b_down, use_reentrant=False)

        # ── Decoder (each up-block checkpointed) ─────────────────────────────
        curr = b_feat
        for up_block in self.up_blocks:
            skip_feat = skips.pop()

            # closure captures up_block and skip_feat by reference
            def _run_up(inp: torch.Tensor, _up=up_block, _skip=skip_feat) -> torch.Tensor:
                return _up(inp, _skip)

            curr = torch_ckpt.checkpoint(_run_up, curr, use_reentrant=False)

        # ── Output head + land mask (same as forward()) ───────────────────────
        logits_sigmoid = self.outc(curr)

        if self.land_mask is not None:
            if self.land_mask.shape[-2:] != logits_sigmoid.shape[-2:]:
                mask = F.interpolate(
                    self.land_mask,
                    size=logits_sigmoid.shape[-2:],
                    mode="nearest",
                )
            else:
                mask = self.land_mask
            output = logits_sigmoid * mask
        else:
            output = logits_sigmoid

        return output


if __name__ == "__main__":
    from pathlib import Path
    from data_pipeline import SeaIceDataPipeline
    from torch_dataset import create_dataloaders

    print("=" * 75)
    print("Testing SeaIceUNet Model with Pipeline Integration")
    print("=" * 75)

    base_dir = Path(__file__).parent / "data"
    sea_ice_pattern = str(base_dir / "sea_ice" / "*.nc")
    mask_file = str(base_dir / "land_mask.nc")

    # 1. Load data pipeline
    pipeline = SeaIceDataPipeline(
        data_path=sea_ice_pattern,
        mask_path=mask_file,
    )
    pipeline.load().clean().normalize(max_value=100.0)

    # 2. Extract U-Net formatted sequences & dataloaders
    window_size = 7
    X_unet, Y_unet = pipeline.format_for_model("unet", window_size=window_size, horizon=1)
    train_loader, val_loader = create_dataloaders(
        X_unet, Y_unet, val_split=0.2, batch_size=8, shuffle_train=True
    )

    # 3. Instantiate SeaIceUNet with pipeline's land mask
    land_mask_tensor = torch.from_numpy(pipeline.land_mask)
    model = SeaIceUNet(
        in_channels=window_size,
        out_channels=1,
        base_filters=32,
        depth=4,
        land_mask=land_mask_tensor,
    )

    # 4. Forward pass using real batch from train_loader
    batch_x, batch_y = next(iter(train_loader))
    print(f"\nPulled real train batch:")
    print(f"  Input batch_x shape:  {batch_x.shape} (Batch, Channels, H, W)")
    print(f"  Target batch_y shape: {batch_y.shape} (Batch, 1, H, W)")

    model.eval()
    with torch.no_grad():
        pred_y = model(batch_x)

    print(f"\nForward pass output:")
    print(f"  Output pred_y shape:  {pred_y.shape}")
    print(f"  Matches target shape: {pred_y.shape == batch_y.shape}")
    print(f"  Output value range:   [{pred_y.min().item():.4f}, {pred_y.max().item():.4f}]")

    # 5. Confirm land pixels are exactly 0.0
    land_mask_np = pipeline.land_mask
    land_indices = land_mask_np == 0.0
    pred_y_np = pred_y.squeeze(1).numpy()
    land_values_max = np.max(pred_y_np[:, land_indices])
    print(f"  Max value over land:  {land_values_max:.6f} (must be 0.000000)")
    assert land_values_max == 0.0, "Land pixels were not strictly zeroed!"

    # 6. Parameter count
    total_params = model.count_parameters()
    print(f"\nModel Parameter Count: {total_params:,} trainable parameters")

    # 7. Non-power-of-2 / Odd Grid Dimension Test
    print("\n--- Non-Power-of-2 / Odd Grid Dimension Compatibility Test ---")
    odd_h, odd_w = 67, 53  # Prime/odd grid dimensions
    dummy_odd_x = torch.randn(2, window_size, odd_h, odd_w)
    # Model without land mask (testing pure convolution and padding)
    model_flexible = SeaIceUNet(
        in_channels=window_size,
        out_channels=1,
        base_filters=32,
        depth=4,
        land_mask=None,
    )
    with torch.no_grad():
        odd_out = model_flexible(dummy_odd_x)
    print(f"  Input shape:  {dummy_odd_x.shape}")
    print(f"  Output shape: {odd_out.shape}")
    assert odd_out.shape == (2, 1, odd_h, odd_w), "Odd grid shape mismatch!"
    print("  Odd grid dimension forward pass completed without error!")

    print("\n" + "=" * 75)
    print("SeaIceUNet Verification Complete!")
    print("=" * 75)
