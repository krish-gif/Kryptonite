"""Pytest unit tests for PyTorch SeaIceDataset and chronological DataLoaders."""

import numpy as np
import pytest
import torch

from torch_dataset import SeaIceDataset, chronological_split, create_dataloaders


@pytest.fixture
def dummy_numpy_data():
    """Create dummy arrays for ConvLSTM (5D) and U-Net (4D) testing."""
    n_samples = 30
    h, w = 16, 16
    window_size = 5

    # ConvLSTM shapes: X is (N, T, C, H, W), Y is (N, 1, 1, H, W)
    # Give sample i the constant value i so we can track temporal ordering
    X_conv = np.zeros((n_samples, window_size, 1, h, w), dtype=np.float32)
    Y_conv = np.zeros((n_samples, 1, 1, h, w), dtype=np.float32)
    for i in range(n_samples):
        X_conv[i] = float(i)
        Y_conv[i] = float(i + window_size)

    # U-Net shapes: X is (N, C, H, W), Y is (N, 1, H, W)
    X_unet = np.zeros((n_samples, window_size, h, w), dtype=np.float32)
    Y_unet = np.zeros((n_samples, 1, h, w), dtype=np.float32)
    for i in range(n_samples):
        X_unet[i] = float(i)
        Y_unet[i] = float(i + window_size)

    return {
        "n_samples": n_samples,
        "X_conv": X_conv,
        "Y_conv": Y_conv,
        "X_unet": X_unet,
        "Y_unet": Y_unet,
        "window_size": window_size,
        "h": h,
        "w": w,
    }


def test_dataset_len_and_indexing(dummy_numpy_data):
    """SeaIceDataset.__len__ and __getitem__ behavior."""
    X = dummy_numpy_data["X_conv"]
    Y = dummy_numpy_data["Y_conv"]

    dataset = SeaIceDataset(X, Y)
    assert len(dataset) == dummy_numpy_data["n_samples"]

    # Sample item shape should have NO leading batch dimension
    x_item, y_item = dataset[0]
    assert isinstance(x_item, torch.Tensor)
    assert isinstance(y_item, torch.Tensor)
    assert x_item.shape == (dummy_numpy_data["window_size"], 1, 16, 16)
    assert y_item.shape == (1, 1, 16, 16)


def test_dataset_dtype_enforces_float32():
    """SeaIceDataset must return torch.float32 even if input is float64."""
    X_f64 = np.ones((10, 3, 8, 8), dtype=np.float64)
    Y_f64 = np.ones((10, 1, 8, 8), dtype=np.float64)

    dataset = SeaIceDataset(X_f64, Y_f64)
    x, y = dataset[0]
    assert x.dtype == torch.float32
    assert y.dtype == torch.float32


def test_dataset_invalid_inputs_raise():
    """SeaIceDataset must raise clear errors for invalid inputs."""
    with pytest.raises(TypeError, match="X must be a numpy.ndarray"):
        SeaIceDataset([1, 2, 3], np.zeros((3, 2)))

    with pytest.raises(TypeError, match="Y must be a numpy.ndarray"):
        SeaIceDataset(np.zeros((3, 2)), [1, 2, 3])

    with pytest.raises(ValueError, match="Sample count mismatch"):
        SeaIceDataset(np.zeros((5, 2)), np.zeros((4, 2)))

    with pytest.raises(ValueError, match="cannot be initialized with 0 samples"):
        SeaIceDataset(np.empty((0, 2)), np.empty((0, 2)))


def test_chronological_split_strict_ordering(dummy_numpy_data):
    """Validation samples must be strictly later in time than training samples."""
    X = dummy_numpy_data["X_conv"]
    Y = dummy_numpy_data["Y_conv"]
    n_samples = dummy_numpy_data["n_samples"]  # 30

    val_split = 0.2
    X_train, Y_train, X_val, Y_val = chronological_split(X, Y, val_split=val_split)

    # 1. Sum of split lengths must match total samples
    assert len(X_train) + len(X_val) == n_samples
    assert len(Y_train) + len(Y_val) == n_samples
    assert len(X_val) == int(round(n_samples * val_split))  # 6 val samples, 24 train samples

    # 2. Strict chronological order: values in train are 0..23, values in val are 24..29
    assert np.max(X_train) < np.min(X_val), "Data leakage! Train max is not strictly < Val min"
    assert np.max(Y_train) < np.min(Y_val), "Data leakage! Train target max is not strictly < Val target min"


def test_chronological_split_invalid_val_split():
    """chronological_split must validate split fractions and edge cases."""
    X = np.ones((10, 2))
    Y = np.ones((10, 1))

    with pytest.raises(ValueError, match="strictly between 0.0 and 1.0"):
        chronological_split(X, Y, val_split=0.0)

    with pytest.raises(ValueError, match="strictly between 0.0 and 1.0"):
        chronological_split(X, Y, val_split=1.0)

    # Dataset with 3 samples and val_split=0.01 -> val_size = round(0.03) = 0
    X_small = np.ones((3, 2))
    Y_small = np.ones((3, 1))
    with pytest.raises(ValueError, match="produces 0 validation samples"):
        chronological_split(X_small, Y_small, val_split=0.01)

    # val_split=0.99 on 3 samples -> val_size = 3 -> train_size = 0
    with pytest.raises(ValueError, match="produces 0 training samples"):
        chronological_split(X_small, Y_small, val_split=0.99)


def test_create_dataloaders_convlstm(dummy_numpy_data):
    """create_dataloaders with ConvLSTM shapes."""
    X = dummy_numpy_data["X_conv"]
    Y = dummy_numpy_data["Y_conv"]

    batch_size = 8
    train_loader, val_loader = create_dataloaders(
        X, Y, val_split=0.2, batch_size=batch_size, shuffle_train=True
    )

    assert len(train_loader.dataset) == 24
    assert len(val_loader.dataset) == 6

    # Test batch extraction
    batch_x, batch_y = next(iter(train_loader))
    assert batch_x.dtype == torch.float32
    assert batch_y.dtype == torch.float32
    # Leading dimension must equal batch_size
    assert batch_x.shape == (batch_size, dummy_numpy_data["window_size"], 1, 16, 16)
    assert batch_y.shape == (batch_size, 1, 1, 16, 16)

    # Full iteration without shape errors
    train_samples_seen = sum(bx.shape[0] for bx, _ in train_loader)
    val_samples_seen = sum(bx.shape[0] for bx, _ in val_loader)
    assert train_samples_seen == 24
    assert val_samples_seen == 6


def test_create_dataloaders_unet(dummy_numpy_data):
    """create_dataloaders with U-Net shapes."""
    X = dummy_numpy_data["X_unet"]
    Y = dummy_numpy_data["Y_unet"]

    batch_size = 8
    train_loader, val_loader = create_dataloaders(
        X, Y, val_split=0.2, batch_size=batch_size, shuffle_train=False
    )

    batch_x, batch_y = next(iter(train_loader))
    assert batch_x.dtype == torch.float32
    assert batch_y.dtype == torch.float32
    # In U-Net, channels = window_size = 5
    assert batch_x.shape == (batch_size, dummy_numpy_data["window_size"], 16, 16)
    assert batch_y.shape == (batch_size, 1, 16, 16)

    # Full iteration
    for _ in train_loader:
        pass
    for _ in val_loader:
        pass


def test_create_dataloaders_invalid_batch_size(dummy_numpy_data):
    """batch_size < 1 must raise ValueError."""
    with pytest.raises(ValueError, match="batch_size must be >= 1"):
        create_dataloaders(dummy_numpy_data["X_conv"], dummy_numpy_data["Y_conv"], batch_size=0)
