import torch
import numpy as np
import matplotlib.pyplot as plt

def visualize_brats_prediction(pred, slice_idx=None, auto_find_tumor=True, return_fig=False):
    """
    Visualize a BraTS prediction mask (4-class segmentation) for a single sample.
    
    Args:
        pred: torch.Tensor of shape [1, 4, D, H, W] (logits or probabilities)
        slice_idx: specific slice index to visualize
        auto_find_tumor: if True, automatically find slice with most tumor
        return_fig: if True, returns the matplotlib figure instead of showing it
    Returns:
        fig: Matplotlib figure if return_fig=True
    """
    # Move to CPU and convert to numpy
    pred = pred.cpu().numpy()  # [1, 4, D, H, W]
    
    # Convert logits/probabilities to class labels
    seg = np.argmax(pred[0], axis=0)  # [D, H, W]
    
    depth = seg.shape[0]
    
    # Determine slice to visualize
    if slice_idx is not None:
        slice_idx_use = min(slice_idx, depth - 1)
    elif auto_find_tumor:
        tumor_counts = [np.sum(seg[d] > 0) for d in range(depth)]
        slice_idx_use = np.argmax(tumor_counts)
        if tumor_counts[slice_idx_use] == 0:
            slice_idx_use = depth // 2
            print(f"⚠️  No tumor found in any slice, showing middle slice")
    else:
        slice_idx_use = depth // 2
    
    # Plot the segmentation slice
    seg_slice = seg[slice_idx_use, :, :]
    
    fig, ax = plt.subplots(figsize=(6, 6))
    cmap = plt.cm.colors.ListedColormap(['black', 'red', 'green', 'blue'])
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5]
    norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)
    
    im = ax.imshow(seg_slice, cmap=cmap, norm=norm)
    ax.set_title(f'Segmentation Slice {slice_idx_use}/{depth}', fontsize=14)
    ax.axis('off')
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, ticks=[0, 1, 2, 3])
    cbar.ax.set_yticklabels(['BG', 'NCR', 'ED', 'ET'])
    
    # Print statistics
    print(f"\nSegmentation class distribution:")
    unique, counts = np.unique(seg, return_counts=True)
    total_voxels = seg.size
    class_names = {0: 'Background', 1: 'NCR', 2: 'ED', 3: 'ET'}
    for class_id, count in zip(unique, counts):
        percentage = (count / total_voxels) * 100
        print(f"  Class {int(class_id)} ({class_names.get(int(class_id), 'Unknown')}): "
              f"{count} voxels ({percentage:.2f}%)")
    
    if return_fig:
        return fig
    else:
        plt.show()
