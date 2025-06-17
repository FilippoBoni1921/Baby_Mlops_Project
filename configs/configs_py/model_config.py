from dataclasses import dataclass

@dataclass
class ModelConfig:
    name: str = "google/bert_uncased_L-2_H-128_A-2"
    tokenizer: str = "google/bert_uncased_L-2_H-128_A-2"
