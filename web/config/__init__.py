import os

BASE_DIR = os.path.abspath(os.path.dirname(__package__))

UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
RESULT_FOLDER = os.path.join(BASE_DIR, "static", "results")
MODEL_PATH = os.path.join(BASE_DIR, "model", "transunet_model.pth")

MAX_FILE_SIZE = 200000
ALLOWED_FORMATS = ["nii", "nii.gz", "png"]

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)