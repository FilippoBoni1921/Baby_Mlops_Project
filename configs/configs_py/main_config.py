from dataclasses import dataclass
from model_config import ModelConfig
from training_config import TrainingConfig
from processing_config import ProcessingConfig

@dataclass
class Config:
    model: ModelConfig = ModelConfig()
    training: TrainingConfig = TrainingConfig()
    processing: ProcessingConfig = ProcessingConfig()
