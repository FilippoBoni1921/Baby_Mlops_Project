from dataclasses import dataclass

@dataclass
class TrainingConfig:
    batch_size: int = 32
    lr: float = 0.001
    epochs: int = 10
