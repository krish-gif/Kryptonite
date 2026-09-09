"""Visual Check for Sea-Ice Data Pipeline.

Loads cached PyTorch tensors and plots one complete sequence
(N input days and K target days) to visually verify that the data
looks geographically correct and the sequence makes sense.
"""

import os
import torch
import matplotlib.pyplot as plt

def plot_sequence(cache_dir: str, save_path: str, sequence_idx: int = 0):
    """Load cached tensors and plot the specified sequence."""
    
    print(f"Loading cached tensors from {cache_dir}...")
    try:
        X_train = torch.load(os.path.join(cache_dir, "X_train.pt"), weights_only=True)
        Y_train = torch.load(os.path.join(cache_dir, "Y_train.pt"), weights_only=True)
    except FileNotFoundError as e:
        print(f"Error: Could not find cached data: {e}")
        print("Make sure you run data_cache.py first!")
        return

    # Ensure sequence index is valid
    if sequence_idx >= X_train.shape[0]:
        print(f"Error: Sequence index {sequence_idx} out of bounds (max {X_train.shape[0]-1})")
        return
        
    # Get the sequence (N, C, H, W) and (K, C, H, W)
    x_seq = X_train[sequence_idx]
    y_seq = Y_train[sequence_idx]
    
    N = x_seq.shape[0]
    K = y_seq.shape[0]
    
    print(f"Plotting sequence {sequence_idx}: {N} input frames, {K} target frames...")
    
    # Create a plot with N+K columns
    fig, axes = plt.subplots(1, N + K, figsize=(3 * (N + K), 4))
    
    # If axes is not an array (e.g., N+K=1), make it an array
    if not isinstance(axes, (list, np.ndarray)):
        axes = [axes]
        
    # Plot Input frames
    for i in range(N):
        ax = axes[i]
        # X is (N, C, H, W), we plot the first channel [C=0]
        im = ax.imshow(x_seq[i, 0].numpy(), cmap='Blues_r', vmin=0, vmax=1)
        ax.set_title(f"Input Day {i+1}")
        ax.axis('off')
        
    # Plot Target frames
    for j in range(K):
        ax = axes[N + j]
        im = ax.imshow(y_seq[j, 0].numpy(), cmap='Blues_r', vmin=0, vmax=1)
        ax.set_title(f"Target Day {j+1}")
        # Add a distinct border for target frames to separate them
        for spine in ax.spines.values():
            spine.set_edgecolor('red')
            spine.set_linewidth(2)
        ax.set_xticks([])
        ax.set_yticks([])

    # Add colorbar
    cbar_ax = fig.add_axes([0.92, 0.25, 0.01, 0.5]) # [left, bottom, width, height]
    fig.colorbar(im, cax=cbar_ax, label='Sea Ice Concentration')
    
    plt.suptitle(f"Sea-Ice Sequence {sequence_idx} (Blue=Ice, White/Black=No Ice/Land)", fontsize=16)
    
    # Save the plot
    out_dir = os.path.dirname(save_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    print(f"Saved visual validation plot to: {os.path.abspath(save_path)}")
    plt.close()

if __name__ == "__main__":
    # Local paths
    import numpy as np # Needed if axes fallback hits
    cache_directory = "cache"
    output_png = "visual_check.png"
    
    plot_sequence(cache_dir=cache_directory, save_path=output_png, sequence_idx=0)
