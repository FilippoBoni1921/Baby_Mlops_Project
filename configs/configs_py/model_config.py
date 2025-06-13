from dataclasses import dataclass

@dataclass
class ModelConfig:
    name: str = "resnet50"
    tokenizer: str = "bert-base"
