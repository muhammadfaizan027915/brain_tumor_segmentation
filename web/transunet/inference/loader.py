import torch
import torchio as tio
import glob
import os

STANDARD_MODALITY_ORDER = ['t1n', 't1c', 't2w', 't2f']


def sort_modalities(paths):
    sorted_list = []
    for tag in STANDARD_MODALITY_ORDER:
        match = next((p for p in paths if f"-{tag}.nii" in p.lower()), None)
        sorted_list.append(match)
    return sorted_list


def get_preprocessing_transform(config):
    transforms = []
    if config.patch_size:
        transforms.append(tio.CropOrPad(config.patch_size))

    if config.normalize_method == "z-score":
        transforms.append(tio.ZNormalization(masking_method=tio.ZNormalization.mean))
    elif config.normalize_method == "min-max":
        transforms.append(tio.RescaleIntensity(out_min_max=(0, 1)))

    return tio.Compose(transforms)


def load_case_folder(case_folder: str):
    modality_paths = glob.glob(os.path.join(case_folder, "*.nii"))
    sorted_mods = sort_modalities(modality_paths)

    if any(p is None for p in sorted_mods):
        raise RuntimeError(f"Missing one or more required modalities in {case_folder}")

    return sorted_mods
