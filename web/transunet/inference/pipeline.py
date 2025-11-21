import torch
import torchio as tio
from transunet.inference.loader import load_case_folder, get_preprocessing_transform


def build_subject(mod_paths):
    return tio.Subject(
        t1n=tio.ScalarImage(mod_paths[0]),
        t1c=tio.ScalarImage(mod_paths[1]),
        t2w=tio.ScalarImage(mod_paths[2]),
        t2f=tio.ScalarImage(mod_paths[3]),
    )


def subject_to_tensor(subject):
    # subject.*.data has shape [1, H, W, D] -> squeeze C dim later, but we concat channels first
    img = torch.cat([
        subject.t1n.data,
        subject.t1c.data,
        subject.t2w.data,
        subject.t2f.data,
    ], dim=0)  # [4, H, W, D]

    # Convert to [4, D, H, W] for model
    img = img.permute(0, 3, 1, 2).contiguous()
    return img.unsqueeze(0)   # [1, 4, D, H, W]


@torch.inference_mode()
def predict_case(case_folder: str, model, data_config, device):
    mod_paths = load_case_folder(case_folder)

    # keep a raw copy (no transforms) — useful to save original images / affines
    raw_subject = build_subject(mod_paths)

    # apply preprocessing transforms (CropOrPad, normalization, etc.)
    transform = get_preprocessing_transform(data_config)
    processed_subject = transform(raw_subject)

    # build input tensor from processed subject
    img = subject_to_tensor(processed_subject)
    img = img.to(device)

    pred = model(img)

    # return prediction + processed subject (used for visualization) + raw subject (if needed)
    return pred.cpu(), processed_subject, raw_subject