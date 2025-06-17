from dataclasses import dataclass

@dataclass
class ProcessingConfig:
    batch_size: int = 128
    max_length: int = 128
