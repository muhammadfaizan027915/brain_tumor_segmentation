import torch
import pathlib
from config import MODEL_PATH
from core.models.segmentation_result import SegmentationResult
from transunet.configs import DataConfig, ModelConfig
from transunet.inference import predict_case
from transunet.model import TransUNet3D

pathlib.PosixPath = pathlib.WindowsPath


class SegmentationPredictor:
    def __init__(self):
        self.model_path = MODEL_PATH
        self.use_gpu = torch.cuda.is_available()
        self.data_config = DataConfig()
        self.model_config = ModelConfig()

    def run_transunet(self):
        model = TransUNet3D(self.model_config)
        checkpoint = torch.load(
            self.model_path, map_location="cpu", weights_only=False)

        model.load_state_dict(checkpoint["model_state_dict"])
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        model.eval()

        return model

    def predict(self, case_path, out_path):
        pathlib.Path(out_path).mkdir(parents=True, exist_ok=True)
        device = "cuda" if self.use_gpu else "cpu"
        model = self.run_transunet()

        pred, processed_subject, raw_subject = predict_case(
            case_path, model, self.data_config, device
        )

        segmentation_result = SegmentationResult(
            source_path=case_path,
            subject=processed_subject,
            raw_mask=pred,
            metadata=None
        )

        torch.save(
            {"pred": pred, "subject": raw_subject},
            f"{out_path}/prediction_with_subject.pt"
        )

        return segmentation_result
