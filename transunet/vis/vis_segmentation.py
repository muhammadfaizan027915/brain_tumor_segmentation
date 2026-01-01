import numpy as np
import matplotlib.pyplot as plt

def _to_DHW(arr, ref_shape):
    """
    Ensure `arr` has shape [D, H, W]. `arr` typically is [H, W, D].
    We'll try the common permutations and return an array shaped [D, H, W].
    """
    if arr.shape == ref_shape:
        return arr  # already [D,H,W]
    # common case: arr is [H, W, D] -> transpose to [D, H, W]
    if arr.shape == (ref_shape[1], ref_shape[2], ref_shape[0]):
        return np.transpose(arr, (2, 0, 1))
    # try (W, D, H) or other permutations as a last attempt
    for perm in [(2, 0, 1), (1, 2, 0), (0, 2, 1), (1, 0, 2), (2, 1, 0)]:
        candidate = np.transpose(arr, perm)
        if candidate.shape == ref_shape:
            return candidate
    # fallback: force reshape if sizes match total voxels (dangerous)
    if arr.size == np.prod(ref_shape):
        return arr.flatten().reshape(ref_shape)
    raise RuntimeError(f"Can't convert array shape {arr.shape} to target {ref_shape}")


def visualize_brats_subject_pred(pred, processed_subject, raw_subject=None,
                                 slice_idx=None, auto_find_tumor=True, normalize=True,
                                 return_fig=False):
    """
    pred: torch.Tensor [1, C, D, H, W]  (logits or probs)
    processed_subject: TorchIO Subject after transforms (used to build model input)
    raw_subject: optional TorchIO Subject before transforms (to save or inspect originals)
    return_fig: if True, return matplotlib Figure instead of showing
    """
    pred_np = pred.cpu().numpy()[0]           # [C, D, H, W]
    seg = np.argmax(pred_np, axis=0)          # [D, H, W]
    D, H, W = seg.shape
    ref_shape = (D, H, W)

    modalities = ['t1n', 't1c', 't2w', 't2f']
    images = []
    for mod in modalities:
        arr = processed_subject[mod].data.squeeze(0).numpy()  # usually [H, W, D]
        arr_dhw = _to_DHW(arr, ref_shape)                    # now [D, H, W]
        images.append(arr_dhw)

    images = np.array(images)  # [4, D, H, W]

    # choose slice
    if slice_idx is not None:
        slice_use = min(slice_idx, D - 1)
    elif auto_find_tumor:
        tumor_counts = [np.sum(seg[d] > 0) for d in range(D)]
        slice_use = int(np.argmax(tumor_counts))
        if tumor_counts[slice_use] == 0:
            slice_use = D // 2
            print("⚠️  No tumor found, showing middle slice")
    else:
        slice_use = D // 2

    # plotting
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    fig.suptitle(f"Slice {slice_use}/{D}", fontsize=16, fontweight='bold')

    for i, mod in enumerate(modalities):
        img_slice = images[i, slice_use]
        if normalize:
            img_slice = (img_slice - img_slice.min()) / (img_slice.max() - img_slice.min() + 1e-8)
        axes[i].imshow(img_slice, cmap='gray', aspect='auto')
        axes[i].set_title(mod)
        axes[i].axis('off')

    ax = axes[4]
    seg_slice = seg[slice_use]
    cmap = plt.cm.colors.ListedColormap(['black', 'red', 'green', 'blue'])
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5]
    norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)
    im = ax.imshow(seg_slice, cmap=cmap, norm=norm, aspect='auto')
    ax.set_title("Predicted Segmentation")
    ax.axis('off')
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, ticks=[0,1,2,3])
    cbar.ax.set_yticklabels(['BG', 'NCR', 'ED', 'ET'])

    plt.tight_layout()

    # stats (optional)
    print("\n=== Modality Statistics ===")
    for i, mod in enumerate(modalities):
        arr_full = images[i]
        print(f"{mod:6s}: min={arr_full.min():.4f}, max={arr_full.max():.4f}, mean={arr_full.mean():.4f}, std={arr_full.std():.4f}")

    print("\n=== Predicted Segmentation Distribution ===")
    unique, counts = np.unique(seg, return_counts=True)
    total_voxels = seg.size
    class_names = {0:'BG',1:'NCR',2:'ED',3:'ET'}
    for u,c in zip(unique, counts):
        pct = (c/total_voxels)*100
        print(f"Class {u} ({class_names.get(u,'Unknown')}): {c} voxels ({pct:.2f}%)")

    if return_fig:
        return fig
    else:
        plt.show()
