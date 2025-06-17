from dataclasses import dataclass

@dataclass
class TrainingConfig:
    batch_size: int = 256
    lr: float = 0.01
    epochs: int = 5
