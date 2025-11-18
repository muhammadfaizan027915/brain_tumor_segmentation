import os
from config import UPLOAD_FOLDER
from core.models.mri_file import MRIFile

class FileUploadHandler:
    @staticmethod
    def store(file: MRIFile) -> str:
        filepath = os.path.join(UPLOAD_FOLDER, file.name)
        with open(filepath, "wb") as f:
            f.write(file.raw_data)
        return filepath
    
    @staticmethod
    def retrieve(filepath: str) -> bytes:
        with open(filepath, "rb") as f:
            return f.read()
    
    @staticmethod
    def delete(filepath: str) -> bool:
        if os.path.exists(filepath):
            os.remove(filepath)
            return True
        return False
