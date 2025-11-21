import torch
import argparse
import pathlib
pathlib.PosixPath = pathlib.WindowsPath

from transunet.configs import DataConfig, ModelConfig

model_config = ModelConfig()   
data_config  = DataConfig()

from model import TransUNet3D
from inference import predict_case

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=str, required=True)
    args = parser.parse_args()

    case_path = f"data/uploads/{args.case}"
    out_path  = f"data/predictions/{args.case}"
    pathlib.Path(out_path).mkdir(parents=True, exist_ok=True)

    # configs
    data_config  = DataConfig()
    model_config = ModelConfig()

    # model
    model = TransUNet3D(model_config)
    checkpoint = torch.load("checkpoints/transunet.pth", map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    # pred, subject = predict_case(case_path, model, data_config, device)

    # save prediction
    # torch.save(pred, f"{out_path}/prediction.pt")
    # print(f"Prediction saved → {out_path}/prediction.pt")

    pred, processed_subject, raw_subject = predict_case(case_path, model, data_config, device)

    # Save the predictions and processed subject
    torch.save({
        "pred": pred,
        "subject": processed_subject
    }, f"{out_path}/prediction_with_subject.pt")

    # Print confirmation
    print(f"Predictions and processed subject have been saved to: {out_path}/prediction_with_subject.pt")



if __name__ == "__main__":
    main()

