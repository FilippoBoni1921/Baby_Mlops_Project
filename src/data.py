import src.constants as constants

from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer

class DatasetLoader:
    def __init__(self, model_name=constants.BASE_MODEL, batch_size=32, max_length=512):
        self.batch_size = batch_size
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize_data(self, example):
        return self.tokenizer(
            example[constants.SENTENCE],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
        )

    def prepare(self):
        # Load dataset
        dataset = load_dataset(constants.GLUE, constants.COLA)

        # Tokenize
        dataset = dataset.map(self.tokenize_data, batched=True)

        # Set format for PyTorch
        dataset.set_format(
            type="torch",
            columns=[constants.INPUT_IDS, constants.ATTENTION_MASK, constants.LABEL]
        )

        # Split into train/val
        self.train_dataset = dataset[constants.TRAIN]
        self.val_dataset = dataset[constants.VALIDATION]

    def get_dataloaders(self):
        train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)
        return train_loader, val_loader

if __name__ == "__main__":
    data_module = DatasetLoader()
    data_module.prepare()
    train_loader, val_loader = data_module.get_dataloaders()
    
    batch = next(iter(train_loader))
    print("Input IDs shape:", batch[constants.INPUT_IDS].shape)
