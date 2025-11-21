import torch
import numpy as np
import matplotlib.pyplot as plt

from transunet.vis.vis_segmentation import _to_DHW


def visualize_brats_overlay(pred, processed_subject, modality='t1c',
                            slice_idx=None, auto_find_tumor=True,
                            normalize=True, return_fig=False):
    """
    Overlay predicted segmentation on a chosen modality slice.
    
    pred: torch.Tensor [1, C, D, H, W]
    processed_subject: TorchIO Subject after transforms
    modality: one of 't1n','t1c','t2w','t2f' to overlay segmentation on
    slice_idx: optional slice along depth
    auto_find_tumor: select slice with largest tumor area if True
    normalize: normalize image for display
    return_fig: if True, returns matplotlib Figure instead of showing
    """
    # Convert prediction to segmentation mask [D, H, W]
    pred_np = pred.cpu().numpy()[0]       # [C, D, H, W]
    seg = np.argmax(pred_np, axis=0)      # [D, H, W]
    D, H, W = seg.shape
    ref_shape = (D, H, W)

    # Convert chosen modality to [D, H, W]
    x = processed_subject[modality]

    # Convert to numpy if tensor
    if torch.is_tensor(x):
        arr = x.detach().cpu().squeeze(0).numpy()  # remove batch dim
    else:
        arr = np.squeeze(x)  # handle numpy array with singleton dims

    img = _to_DHW(arr, ref_shape)

    # Choose slice
    if slice_idx is not None:
        slice_use = min(slice_idx, D - 1)
    elif auto_find_tumor:
        tumor_counts = [np.sum(seg[d] > 0) for d in range(D)]
        slice_use = int(np.argmax(tumor_counts))
        if tumor_counts[slice_use] == 0:
            slice_use = D // 2
            print("⚠️ No tumor found, showing middle slice")
    else:
        slice_use = D // 2

    img_slice = img[slice_use]
    seg_slice = seg[slice_use]

    # Normalize image
    if normalize:
        img_slice = (img_slice - img_slice.min()) / (img_slice.max() - img_slice.min() + 1e-8)

    # Create figure
    fig, ax = plt.subplots(figsize=(6,6))
    ax.imshow(img_slice, cmap='gray')
    
    # Overlay segmentation
    cmap = plt.cm.get_cmap('jet', 4)  # 4 classes: BG,NCR,ED,ET
    seg_display = np.ma.masked_where(seg_slice==0, seg_slice)  # mask background
    ax.imshow(seg_display, cmap=cmap, alpha=0.5, interpolation='none')

    ax.axis('off')
    ax.set_title(f"{modality} with Segmentation Overlay (Slice {slice_use}/{D})")

    if return_fig:
        return fig
    else:
        plt.show()
