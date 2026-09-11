"""Loss Functions and Evaluation Metrics for Sea-Ice Forecasting.

Provides:
1. `SeaIceLoss`: Differentiable, land-masked, boundary-weighted training loss (MSE/MAE)
   with masked denominator normalization for gradient backpropagation.
2. `compute_iiee`: Non-differentiable Integrated Ice Edge Error (IIEE) evaluation metric
   strictly executed under `torch.no_grad()` for validation monitoring.
"""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


logger = logging.getLogger("SeaIceLoss")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _formatter = logging.Formatter(
        "[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    _handler.setFormatter(_formatter)
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)


class SeaIceLoss(nn.Module):
    """Differentiable, land-masked, boundary-weighted loss for sea-ice concentration.

    Calculates squared (MSE) or absolute (MAE) error exclusively over ocean pixels,
    multiplying prediction errors by a detached ice-edge weight map derived from the target.
    The denominator masks out land pixels so the gradient signal is not diluted by static land masses.

    Mathematical Formulation:
        Let M_ocean be the binary ocean mask (1=ocean, 0=land).
        Let target be y, prediction be p.
        Ice edge condition: |y - edge_threshold| <= edge_band (and M_ocean == 1).
        Weight W:
            W = edge_weight  if edge pixel in ocean
            W = 1.0          if non-edge ocean pixel
            W = 0.0          if land pixel
        Loss:
            L = sum(W * error(p, y)) / (sum(W) + eps)

    Args:
        land_mask: Spatial land mask tensor (Height, Width) where 1=ocean, 0=land.
        base_loss: Underlying distance metric: 'mse' or 'mae' (default: 'mse').
        edge_weight: Multiplier applied to marginal ice zone / edge pixels (default: 3.0).
        edge_band: Half-width around edge_threshold defining the edge band (default: 0.05).
        edge_threshold: Ice concentration threshold defining the edge (default: 0.15).
        normalize_by: Normalization strategy:
            - 'weights': Divide by the sum of per-pixel weights (weighted expectation).
            - 'ocean_count': Divide by the total count of ocean pixels across the batch.
            Default is 'weights'.
    """

    def __init__(
        self,
        land_mask: torch.Tensor,
        base_loss: Literal["mse", "mae"] = "mse",
        edge_weight: float = 3.0,
        edge_band: float = 0.05,
        edge_threshold: float = 0.15,
        normalize_by: Literal["weights", "ocean_count"] = "weights",
    ) -> None:
        super().__init__()
        base_key = base_loss.strip().lower()
        if base_key not in ("mse", "mae"):
            raise ValueError(f"Unsupported base_loss '{base_loss}'. Choose 'mse' or 'mae'.")

        if edge_weight < 1.0:
            raise ValueError(f"edge_weight must be >= 1.0, got {edge_weight}")
        if edge_band <= 0.0:
            raise ValueError(f"edge_band must be strictly positive, got {edge_band}")
        if not (0.0 <= edge_threshold <= 1.0):
            raise ValueError(f"edge_threshold must be in [0.0, 1.0], got {edge_threshold}")
        if normalize_by not in ("weights", "ocean_count"):
            raise ValueError(
                f"normalize_by must be 'weights' or 'ocean_count', got {normalize_by}"
            )

        self.base_loss = base_key
        self.edge_weight = float(edge_weight)
        self.edge_band = float(edge_band)
        self.edge_threshold = float(edge_threshold)
        self.normalize_by = normalize_by

        # Register land mask as non-trainable persistent buffer
        if not isinstance(land_mask, torch.Tensor):
            land_mask = torch.tensor(land_mask, dtype=torch.float32)
        else:
            land_mask = land_mask.to(dtype=torch.float32)

        # Standardize mask shape to (1, 1, H, W)
        if land_mask.ndim == 2:
            land_mask = land_mask.unsqueeze(0).unsqueeze(0)
        elif land_mask.ndim == 3:
            land_mask = land_mask.unsqueeze(0)

        # Ensure binary mask: 1.0 for ocean, 0.0 for land
        land_mask = (land_mask != 0.0).float()

        # Check for degenerate case (zero ocean pixels)
        ocean_pixel_count = int((land_mask == 1.0).sum().item())
        if ocean_pixel_count == 0:
            raise ValueError(
                "Degenerate land_mask: mask contains 0 ocean pixels (entire domain is land). "
                "Cannot compute ocean-masked loss."
            )

        self.register_buffer("land_mask", land_mask)
        logger.info(
            f"SeaIceLoss initialized: base_loss={self.base_loss}, edge_weight={self.edge_weight}, "
            f"edge_band=+/-{self.edge_band}, edge_threshold={self.edge_threshold}, "
            f"normalize_by={self.normalize_by} -> Ocean pixels in domain: {ocean_pixel_count:,}"
        )

    def get_boundary_weight_map(self, target: torch.Tensor) -> torch.Tensor:
        """Construct per-pixel boundary weight map derived strictly from the target.

        This map is completely detached from the autograd graph (no gradients flow
        through the weight calculation, only through the prediction error).

        Args:
            target: Ground truth sea-ice concentration of shape (Batch, 1, Height, Width).

        Returns:
            weights: Weight tensor of identical shape with requires_grad=False.
        """
        # Detach target explicitly to ensure no autograd tracking
        target_detached = target.detach()

        # Align land mask if target spatial dimensions differ
        if self.land_mask.shape[-2:] != target.shape[-2:]:
            mask = F.interpolate(
                self.land_mask,
                size=target.shape[-2:],
                mode="nearest",
            )
        else:
            mask = self.land_mask

        ocean_mask = (mask == 1.0)

        # Ice edge condition: target in [edge_threshold - edge_band, edge_threshold + edge_band]
        lower_bound = self.edge_threshold - self.edge_band
        upper_bound = self.edge_threshold + self.edge_band

        is_edge = (
            (target_detached >= lower_bound)
            & (target_detached <= upper_bound)
            & ocean_mask
        )

        # Base ocean weight = 1.0, land weight = 0.0
        weights = torch.where(
            ocean_mask,
            torch.ones_like(target_detached),
            torch.zeros_like(target_detached),
        )

        # Apply edge weight multiplier
        weights = torch.where(
            is_edge,
            torch.full_like(target_detached, self.edge_weight),
            weights,
        )

        return weights

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute differentiable, ocean-masked, boundary-weighted scalar loss.

        Args:
            pred: Predicted concentration tensor of shape (Batch, 1, Height, Width).
            target: Ground truth concentration tensor of shape (Batch, 1, Height, Width).

        Returns:
            Scalar tensor loss ready for .backward().
        """
        if pred.shape != target.shape:
            raise ValueError(
                f"Shape mismatch: pred has shape {pred.shape}, but target has shape {target.shape}"
            )

        # 1. Compute per-pixel raw error (MSE or MAE)
        if self.base_loss == "mse":
            raw_error = (pred - target) ** 2
        else:
            raw_error = torch.abs(pred - target)

        # 2. Derive detached boundary weight map
        weights = self.get_boundary_weight_map(target)

        # 3. Weighted per-pixel error (land pixels have weight 0.0)
        weighted_error = raw_error * weights

        # 4. Masked denominator reduction
        # Sum only over ocean pixels / weights so land fraction does not dilute the gradient
        if self.normalize_by == "weights":
            denom = weights.sum()
        else:  # 'ocean_count'
            # Total ocean pixels across batch
            if self.land_mask.shape[-2:] != target.shape[-2:]:
                mask = F.interpolate(
                    self.land_mask,
                    size=target.shape[-2:],
                    mode="nearest",
                )
            else:
                mask = self.land_mask
            denom = (mask == 1.0).sum() * pred.shape[0]

        # Prevent division by zero with small epsilon
        loss = weighted_error.sum() / (denom + 1e-8)
        return loss


@torch.no_grad()
def compute_iiee(
    pred: torch.Tensor,
    target: torch.Tensor,
    land_mask: torch.Tensor,
    threshold: float = 0.15,
) -> float:
    """Compute non-differentiable Integrated Ice Edge Error (IIEE) for validation reporting.

    IIEE (Goessling et al. 2016) measures the area where forecast and observation disagree
    on whether concentration is above or below a threshold (conventionally 15% / 0.15).
    Because it uses indicator step functions, it is non-differentiable and must NEVER
    be used as the backpropagation loss.

    Formula:
        Disagreement = (pred > threshold) != (target > threshold) over ocean pixels.
        IIEE = Count(Disagreement) / Count(Ocean_Pixels_in_Batch)

    Args:
        pred: Predicted concentration tensor of shape (Batch, 1, Height, Width).
        target: Ground truth concentration tensor of shape (Batch, 1, Height, Width).
        land_mask: Land mask tensor (1=ocean, 0=land).
        threshold: Ice concentration threshold defining the ice edge (default: 0.15).

    Returns:
        float: Fraction of ocean pixels with edge disagreement in [0.0, 1.0].
               Returned value is a pure Python float with NO autograd graph.
    """
    if pred.shape != target.shape:
        raise ValueError(
            f"Shape mismatch: pred shape {pred.shape} != target shape {target.shape}"
        )

    # Standardize land mask to match spatial dimensions
    if not isinstance(land_mask, torch.Tensor):
        land_mask_t = torch.tensor(land_mask, dtype=torch.float32, device=pred.device)
    else:
        land_mask_t = land_mask.to(dtype=torch.float32, device=pred.device)

    if land_mask_t.ndim == 2:
        land_mask_t = land_mask_t.unsqueeze(0).unsqueeze(0)
    elif land_mask_t.ndim == 3:
        land_mask_t = land_mask_t.unsqueeze(0)

    if land_mask_t.shape[-2:] != pred.shape[-2:]:
        mask = F.interpolate(
            land_mask_t,
            size=pred.shape[-2:],
            mode="nearest",
        )
    else:
        mask = land_mask_t

    ocean_mask = (mask == 1.0)
    total_ocean_pixels = int(ocean_mask.sum().item()) * pred.shape[0]

    if total_ocean_pixels == 0:
        raise ValueError("Degenerate land_mask: contains zero ocean pixels.")

    # Disagreement indicator strictly over ocean pixels
    pred_ice = (pred > threshold) & ocean_mask
    target_ice = (target > threshold) & ocean_mask

    disagreement = (pred_ice != target_ice) & ocean_mask
    disagreement_count = int(disagreement.sum().item())

    # Return pure Python float
    return float(disagreement_count / total_ocean_pixels)


if __name__ == "__main__":
    from pathlib import Path
    from data_pipeline import SeaIceDataPipeline
    from torch_dataset import create_dataloaders
    from unet_model import SeaIceUNet

    print("=" * 75)
    print("Testing SeaIceLoss and IIEE Metric with Pipeline Integration")
    print("=" * 75)

    base_dir = Path(__file__).parent / "data"
    sea_ice_pattern = str(base_dir / "sea_ice" / "*.nc")
    mask_file = str(base_dir / "land_mask.nc")

    # 1. Load data & create real train batch
    pipeline = SeaIceDataPipeline(
        data_path=sea_ice_pattern,
        mask_path=mask_file,
    )
    pipeline.load().clean().normalize(max_value=100.0)

    X_unet, Y_unet = pipeline.format_for_model("unet", window_size=7, horizon=1)
    train_loader, _ = create_dataloaders(
        X_unet, Y_unet, val_split=0.2, batch_size=8, shuffle_train=True
    )

    batch_x, batch_y = next(iter(train_loader))

    # 2. Instantiate untrained SeaIceUNet with land mask
    land_mask_tensor = torch.from_numpy(pipeline.land_mask)
    model = SeaIceUNet(
        in_channels=7,
        out_channels=1,
        base_filters=32,
        depth=4,
        land_mask=land_mask_tensor,
    )
    model.train()

    # Forward pass
    pred_y = model(batch_x)

    # 3. Instantiate SeaIceLoss
    criterion = SeaIceLoss(
        land_mask=land_mask_tensor,
        base_loss="mse",
        edge_weight=3.0,
        edge_band=0.05,
        edge_threshold=0.15,
    )

    # 4. Forward loss calculation
    loss = criterion(pred_y, batch_y)
    print(f"\n--- 1. SeaIceLoss Forward Pass ---")
    print(f"Loss scalar value:      {loss.item():.6f}")
    print(f"requires_grad:          {loss.requires_grad}")
    print(f"grad_fn:                {loss.grad_fn}")
    assert loss.ndim == 0, "Loss must be a 0-dim scalar tensor!"
    assert loss.requires_grad is True, "Loss must be differentiable with requires_grad=True!"
    assert loss.grad_fn is not None, "Loss must have a valid grad_fn attached!"

    # 5. Perfect prediction sanity check: L(target, target) == 0.0
    loss_perfect = criterion(batch_y, batch_y)
    print(f"\n--- 2. Sanity Check: L(target, target) ---")
    print(f"Loss on perfect match:  {loss_perfect.item():.8f} (Expected: 0.00000000)")
    assert loss_perfect.item() == 0.0, "Sanity check failed: L(target, target) != 0.0!"

    # 6. Boundary weight map statistics
    weight_map = criterion.get_boundary_weight_map(batch_y)
    # Check for sample 0
    w_sample0 = weight_map[0, 0].cpu().numpy()
    ocean_mask_np = (pipeline.land_mask == 1.0)
    ocean_weights = w_sample0[ocean_mask_np]

    edge_pixels = (ocean_weights == 3.0).sum()
    total_ocean = ocean_mask_np.sum()
    edge_fraction = (edge_pixels / total_ocean) * 100.0

    print(f"\n--- 3. Boundary Weight Map Statistics (Sample 0) ---")
    print(f"Weight min (ocean):     {ocean_weights.min():.2f}")
    print(f"Weight max (edge):      {ocean_weights.max():.2f}")
    print(f"Weight mean (ocean):    {ocean_weights.mean():.4f}")
    print(f"Edge pixel count:       {edge_pixels:,} / {total_ocean:,} ocean pixels ({edge_fraction:.2f}%)")
    assert ocean_weights.min() == 1.0
    assert ocean_weights.max() == 3.0
    assert 0.0 < edge_fraction < 100.0, "Edge weighting is degenerate (flagged 0% or 100%)!"

    # 7. compute_iiee metric
    iiee_score = compute_iiee(pred_y, batch_y, land_mask=land_mask_tensor, threshold=0.15)
    print(f"\n--- 4. IIEE Evaluation Metric ---")
    print(f"IIEE disagreement rate: {iiee_score:.4f} ({iiee_score * 100:.2f}%)")
    print(f"Type:                   {type(iiee_score).__name__}")
    assert isinstance(iiee_score, float)
    assert 0.0 <= iiee_score <= 1.0

    # 8. Land pixel garbage invariance test
    print(f"\n--- 5. Land Pixel Invariance Verification ---")
    # Clone prediction and overwrite land coordinates with extreme garbage values
    corrupted_pred = pred_y.clone()
    land_bool_4d = (land_mask_tensor == 0.0).unsqueeze(0).unsqueeze(0).expand_as(corrupted_pred)
    corrupted_pred[land_bool_4d] = 99999.0

    loss_corrupted = criterion(corrupted_pred, batch_y)
    iiee_corrupted = compute_iiee(corrupted_pred, batch_y, land_mask=land_mask_tensor)

    print(f"Original Loss:          {loss.item():.8f}")
    print(f"Corrupted Land Loss:    {loss_corrupted.item():.8f}")
    print(f"Original IIEE:          {iiee_score:.8f}")
    print(f"Corrupted Land IIEE:    {iiee_corrupted:.8f}")
    np.testing.assert_allclose(loss.item(), loss_corrupted.item(), rtol=1e-6)
    np.testing.assert_allclose(iiee_score, iiee_corrupted, rtol=1e-6)
    print("Land pixel exclusion confirmed! Garbage over land does not alter loss or IIEE.")

    print("\n" + "=" * 75)
    print("SeaIceLoss & IIEE Verification Complete!")
    print("=" * 75)
    