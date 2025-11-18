from core.models.mri_file import MRIFile
from config import MAX_FILE_SIZE, ALLOWED_FORMATS

class MRIFileValidator:
    def __init__(self):
        self.errors = []
        self.supported_formats = ALLOWED_FORMATS
        self.max_file_size = MAX_FILE_SIZE
        
    def validate_format(self, file: MRIFile) -> bool:
        return file.format in self.supported_formats
    
    def validate_file_size(self, file: MRIFile) -> bool:
        return file.size <= self.max_file_size
    
    def validate(self, file: MRIFile) -> bool:
        is_valid_format = self.validate_format(file)
        is_valid_size = self.validate_file_size(file)

        print("format=", file.format)
        
        if(not is_valid_format):
            self.errors.append(f"Unsupported file format: {file.format}")
    
        if(not is_valid_size):
            self.errors.append(f"File size {file.size} exceeds the limit {self.max_file_size}")
        
        return len(self.errors) == 0
        

    def assign_modality(self, file: MRIFile) -> MRIFile:
        if "flair" in file.name.lower():
            file.set_modality("FLAIR")
        elif "t1" in file.name.lower():
            file.set_modality("T1")
        elif "t2" in file.name.lower():
            file.set_modality("T2")
        else:
            file.set_modality("Unknown")

        return file

    def get_errors(self):
        return self.errors