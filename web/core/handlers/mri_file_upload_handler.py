from typing import BinaryIO
from core.models.mri_file import MRIFile
from core.services.mri_file_validator import MRIFileValidator
from core.handlers.file_storage_handler import FileUploadHandler

class MRIFileUploadHandler:
    def __init__(self):
        self.upload_issues = []

    def validate_and_upload(self, file: BinaryIO, session_id: str):
        file_content = file.read()
        
        mri_file = MRIFile(
            name=file.filename,
            format=file.filename.split(".")[-1].lower(),
            raw_data=file_content,
            size=len(file_content)
        )

        validator = MRIFileValidator()
        is_valid = validator.validate(mri_file)

        if not is_valid:
            self.upload_issues = validator.get_errors()
            return None, self.upload_issues

        mri_file_with_modality = validator.assign_modality(mri_file)
        file_path = FileUploadHandler.store(mri_file_with_modality)

        mri_file_with_modality.set_filepath(file_path)

        return mri_file_with_modality, None

    def get_upload_issues(self):
        return self.upload_issues
