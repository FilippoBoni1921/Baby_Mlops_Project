from dataclasses import dataclass

@dataclass
class ProcessingConfig:
    batch_size: int = 32
    lr: float = 0.001
    epochs: int = 10
