from dataclasses import dataclass, field
from configs.configs_py.model_config import ModelConfig
from configs.configs_py.training_config import TrainingConfig
from configs.configs_py.processing_config import ProcessingConfig

@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
