class MRIFile:
    def __init__(
        self,
        name: str,
        format: str,
        raw_data: str,
        size: int,
        filepath: str = None,
        modality: str = None
    ):
        self.name = name
        self.format = format
        self.raw_data = raw_data
        self.size = size
        self.filepath = filepath
        self.modality = modality

    def get_name(self): return self.name
    def get_format(self): return self.format
    def get_raw_data(self): return self.raw_data
    def get_size(self): return self.size
    def get_modality(self): return self.modality
    def get_filepath(self): return self.filepath

    def set_name(self, name): self.name = name
    def set_format(self, format): self.format = format
    def set_raw_data(self, raw_data): self.raw_data = raw_data
    def set_size(self, size): self.size = size
    def set_modality(self, modality): self.modality = modality
    def set_filepath(self, filepath): self.filepath = filepath
