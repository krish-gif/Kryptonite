from pathlib import Path
from data_pipeline import SeaIceDataPipeline
from torch_dataset import create_dataloaders
from unet_model import SeaIceUNet
from loss_functions import SeaIceLoss
from train import train_model
import torch

base_dir = Path('data')
pipeline = SeaIceDataPipeline(
    data_path=str(base_dir / 'sea_ice' / '*.nc'),
    mask_path=str(base_dir / 'land_mask.nc'),
)
pipeline.load().clean().normalize(max_value=100.0)
land_mask = torch.from_numpy(pipeline.land_mask)

X, Y = pipeline.format_for_model('unet', window_size=7, horizon=1)
train_loader, val_loader = create_dataloaders(X, Y, val_split=0.2, batch_size=8)

model = SeaIceUNet(in_channels=7, out_channels=1, base_filters=32, depth=4, land_mask=land_mask)
criterion = SeaIceLoss(land_mask=land_mask, base_loss='mse')

print('=== LIVE TRAINING CHECK ===')
history = train_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=criterion,
    land_mask=land_mask,
    num_epochs=5,
    lr=1e-3,
    patience=10,
    checkpoint_dir='checkpoints_live/',
    seed=42,
)

losses     = [r['train_loss'] for r in history]
val_losses = [r['val_loss']   for r in history]

print()
print('=== RESULTS ===')
print(f'Epochs completed : {len(history)}')
print(f'Train loss trend : {[round(l, 4) for l in losses]}')
print(f'Val   loss trend : {[round(l, 4) for l in val_losses]}')
print(f'Loss decreased   : {losses[-1] < losses[0]}')
status = 'TRAINING OK' if losses[-1] < losses[0] else 'NOT TRAINING'
print(f'STATUS           : {status}')
