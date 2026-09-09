"""ConvLSTM Encoder-Decoder for Sea-Ice Concentration Forecasting.

Implements a ConvLSTM cell from scratch (no built-in PyTorch layer exists)
and stacks it into an encoder-decoder architecture:

    Encoder:  consumes N input frames one at a time through stacked ConvLSTM layers.
    Decoder:  rolls forward K auto-regressive steps, producing one concentration
              frame per step via a 1×1 conv mapping hidden state → single channel.

All architectural parameters (num_layers, hidden_channels, kernel_size) are
configurable via the Config dataclass.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn

from config import Config


logger = logging.getLogger("ConvLSTM")
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("[%(asctime)s] [%(name)s] %(levelname)s — %(message)s"))
    logger.addHandler(_h)
    logger.setLevel(logging.INFO)


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  ConvLSTM Cell (from scratch)
# ═══════════════════════════════════════════════════════════════════════════════

class ConvLSTMCell(nn.Module):
    """A single Convolutional LSTM cell.

    Standard LSTM gating applied with 2D convolutions instead of matrix multiplies:

        combined = Conv(x_t) + Conv(h_{t-1})        # shape: (B, 4*hidden, H, W)
        i, f, o, g = split(combined, 4)              # each (B, hidden, H, W)
        c_t = sigmoid(f) * c_{t-1} + sigmoid(i) * tanh(g)
        h_t = sigmoid(o) * tanh(c_t)

    Two separate Conv2d layers:
        - ``input_conv``:     input channels  → 4 × hidden channels
        - ``recurrent_conv``: hidden channels → 4 × hidden channels

    Padding is set to ``kernel_size // 2`` so spatial dimensions are preserved.

    Args:
        input_channels: Number of channels in the input tensor ``x_t``.
        hidden_channels: Number of channels in the hidden/cell state.
        kernel_size: Size of the convolving kernel.
    """

    def __init__(
        self,
        input_channels: int,
        hidden_channels: int,
        kernel_size: int,
    ) -> None:
        super().__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        padding = kernel_size // 2  # same-padding

        # Input-to-state convolution:  (B, input_ch, H, W) → (B, 4*hidden_ch, H, W)
        self.input_conv = nn.Conv2d(
            in_channels=input_channels,
            out_channels=4 * hidden_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=True,
        )

        # State-to-state (recurrent) convolution: (B, hidden_ch, H, W) → (B, 4*hidden_ch, H, W)
        self.recurrent_conv = nn.Conv2d(
            in_channels=hidden_channels,
            out_channels=4 * hidden_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=False,  # bias already in input_conv
        )

    def forward(
        self,
        x: torch.Tensor,
        h_prev: torch.Tensor,
        c_prev: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One-step forward through the ConvLSTM cell.

        Args:
            x: Input tensor at current timestep, shape ``(B, input_ch, H, W)``.
            h_prev: Previous hidden state, shape ``(B, hidden_ch, H, W)``.
            c_prev: Previous cell state, shape ``(B, hidden_ch, H, W)``.

        Returns:
            h_next: Next hidden state, shape ``(B, hidden_ch, H, W)``.
            c_next: Next cell state, shape ``(B, hidden_ch, H, W)``.
        """
        # Combined gates: input-to-state + state-to-state
        combined = self.input_conv(x) + self.recurrent_conv(h_prev)

        # Split into 4 gates along channel dim
        i, f, o, g = torch.split(combined, self.hidden_channels, dim=1)

        i = torch.sigmoid(i)   # input gate
        f = torch.sigmoid(f)   # forget gate
        o = torch.sigmoid(o)   # output gate
        g = torch.tanh(g)      # cell candidate

        c_next = f * c_prev + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(
        self, batch_size: int, height: int, width: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Create zero-initialised hidden and cell states.

        Returns:
            h0: shape ``(B, hidden_ch, H, W)``
            c0: shape ``(B, hidden_ch, H, W)``
        """
        shape = (batch_size, self.hidden_channels, height, width)
        h0 = torch.zeros(shape, device=device)
        c0 = torch.zeros(shape, device=device)
        return h0, c0


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  Encoder-Decoder ConvLSTM
# ═══════════════════════════════════════════════════════════════════════════════

class ConvLSTMEncoderDecoder(nn.Module):
    """Encoder-decoder architecture built from stacked ConvLSTM cells.

    Architecture overview::

        Encoder (processes N input frames sequentially):
            for t in 0 .. N-1:
                for layer in 0 .. num_layers-1:
                    h[layer], c[layer] = ConvLSTMCell(input_t, h[layer], c[layer])
                    input_t = h[layer]   # output of this layer feeds next layer

        Decoder (auto-regressive K-step roll-forward):
            decoder_input = last encoder output projected through 1×1 conv
            for t in 0 .. K-1:
                for layer in 0 .. num_layers-1:
                    h[layer], c[layer] = ConvLSTMCell(decoder_input, h[layer], c[layer])
                    decoder_input = h[layer]
                output[t] = sigmoid(conv1x1(h[-1]))    # map to [0, 1]
                decoder_input = output[t]               # auto-regressive feed

    Args:
        input_channels: Channels in input frames (1 for SIC).
        hidden_channels: Hidden state channels per ConvLSTM layer.
        kernel_size: Convolution kernel size.
        num_layers: Number of stacked ConvLSTM layers.
        K: Number of decoder output steps (forecast horizon).
    """

    def __init__(
        self,
        input_channels: int = 1,
        hidden_channels: int = 64,
        kernel_size: int = 3,
        num_layers: int = 2,
        K: int = 3,
    ) -> None:
        super().__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.K = K

        # ── Encoder ConvLSTM layers ───────────────────────────────────────────
        self.encoder_cells = nn.ModuleList()
        for layer_idx in range(num_layers):
            in_ch = input_channels if layer_idx == 0 else hidden_channels
            self.encoder_cells.append(
                ConvLSTMCell(in_ch, hidden_channels, kernel_size)
            )

        # ── Decoder ConvLSTM layers ───────────────────────────────────────────
        # Decoder input at first layer comes from the 1×1 conv output (1 channel)
        # but we map it back to hidden_channels through the first decoder cell.
        self.decoder_cells = nn.ModuleList()
        for layer_idx in range(num_layers):
            # First decoder layer takes 1-channel decoder input (the predicted frame)
            # Subsequent layers take hidden_channels from the layer below
            in_ch = input_channels if layer_idx == 0 else hidden_channels
            self.decoder_cells.append(
                ConvLSTMCell(in_ch, hidden_channels, kernel_size)
            )

        # ── Output projection: hidden → 1-channel concentration ──────────────
        self.output_conv = nn.Conv2d(hidden_channels, input_channels, kernel_size=1)

        # ── Weight initialisation ─────────────────────────────────────────────
        self._init_weights()

        # Log parameter count
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(
            f"ConvLSTMEncoderDecoder created: layers={num_layers}, "
            f"hidden={hidden_channels}, kernel={kernel_size}, K={K} → "
            f"{total_params:,} trainable params"
        )

    def _init_weights(self) -> None:
        """Kaiming normal init for conv weights, zeros for biases."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: encode N input frames, decode K forecast frames.

        Args:
            x: Input tensor of shape ``(B, N, C, H, W)`` where ``C=1``.

        Returns:
            predictions: Tensor of shape ``(B, K, C, H, W)`` with values in [0, 1].
        """
        B, N, C, H, W = x.shape
        device = x.device

        # ── Initialise hidden/cell states for all layers ──────────────────────
        h_list: list[torch.Tensor] = []
        c_list: list[torch.Tensor] = []
        for cell in self.encoder_cells:
            h, c = cell.init_hidden(B, H, W, device)
            h_list.append(h)
            c_list.append(c)

        # ── Encoder: process N input frames ───────────────────────────────────
        for t in range(N):
            input_t = x[:, t, :, :, :]  # (B, C, H, W)
            for layer_idx, cell in enumerate(self.encoder_cells):
                h_list[layer_idx], c_list[layer_idx] = cell(
                    input_t, h_list[layer_idx], c_list[layer_idx]
                )
                input_t = h_list[layer_idx]  # pass to next layer

        # ── Decoder: auto-regressive K-step rollout ───────────────────────────
        # Initialise decoder hidden states from encoder's final states
        # (shared references — decoder continues from where encoder left off)
        outputs: list[torch.Tensor] = []

        # First decoder input: project encoder's last hidden state to 1-channel
        decoder_input = torch.sigmoid(self.output_conv(h_list[-1]))  # (B, C, H, W)

        for t in range(self.K):
            input_t = decoder_input
            for layer_idx, cell in enumerate(self.decoder_cells):
                h_list[layer_idx], c_list[layer_idx] = cell(
                    input_t, h_list[layer_idx], c_list[layer_idx]
                )
                input_t = h_list[layer_idx]

            # Map top-layer hidden state → single concentration channel
            pred_t = torch.sigmoid(self.output_conv(h_list[-1]))  # (B, C, H, W)
            outputs.append(pred_t)
            decoder_input = pred_t  # auto-regressive: feed prediction as next input

        # Stack decoder outputs → (B, K, C, H, W)
        predictions = torch.stack(outputs, dim=1)
        return predictions


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  Factory function using Config
# ═══════════════════════════════════════════════════════════════════════════════

def build_model(cfg: Config) -> ConvLSTMEncoderDecoder:
    """Instantiate a ConvLSTMEncoderDecoder from a Config object.

    Args:
        cfg: Pipeline configuration with model hyperparameters.

    Returns:
        model: Untrained ConvLSTMEncoderDecoder instance.
    """
    model = ConvLSTMEncoderDecoder(
        input_channels=1,
        hidden_channels=cfg.hidden_channels,
        kernel_size=cfg.kernel_size,
        num_layers=cfg.num_layers,
        K=cfg.K,
    )
    return model


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  Standalone test
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    cfg = Config()

    # Quick forward-pass shape test
    model = build_model(cfg)
    B, N, C, H, W = 2, cfg.N, 1, 32, 32  # small spatial dims for speed
    dummy = torch.randn(B, N, C, H, W)

    print(f"Input shape:  {dummy.shape}  (B, N, C, H, W)")
    with torch.no_grad():
        out = model(dummy)
    print(f"Output shape: {out.shape}  (B, K, C, H, W)")
    assert out.shape == (B, cfg.K, C, H, W), f"Shape mismatch! {out.shape}"
    print(f"Value range:  [{out.min().item():.4f}, {out.max().item():.4f}] (should be in [0,1])")
    print("Forward pass OK ✓")
