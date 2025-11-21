import os
from config import UPLOAD_FOLDER
from core.models.mri_file import MRIFile


class FileUploadHandler:
    @staticmethod
    def store(session_id: str, upload_dir_id: str, file: MRIFile) -> str:
        target_dir = os.path.join(UPLOAD_FOLDER, session_id, upload_dir_id)
        os.makedirs(target_dir, exist_ok=True)

        filepath = os.path.join(target_dir, file.name)
        with open(filepath, "wb") as f:
            f.write(file.raw_data)
        return filepath

    @staticmethod
    def delete(filepath: str) -> bool:
        if os.path.exists(filepath):
            os.remove(filepath)
            return True
        return False
