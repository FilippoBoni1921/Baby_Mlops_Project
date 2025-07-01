import src.constants as constants
import torch
from src.data import DatasetLoader
from src.model import ColaClassifier

class ColaInference:
    def __init__(self, model_path, model_name=constants.BASE_MODEL, device="cpu"):
        self.device = device
        self.dataloader = DatasetLoader(model_name=model_name)
        self.model = ColaClassifier()
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.to(device)
        self.model.eval()

    def predict(self, sentence):
        tokenized = self.dataloader.tokenize_data({constants.SENTENCE: sentence})

        input_ids = tokenized[constants.INPUT_IDS].to(self.device)
        attention_mask = tokenized[constants.ATTENTION_MASK].to(self.device)

        with torch.no_grad():
            logits = self.model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(logits, dim=1)
            pred_class = torch.argmax(probs, dim=1).item()
            confidence = probs[0][pred_class].item()

        return pred_class, confidence
