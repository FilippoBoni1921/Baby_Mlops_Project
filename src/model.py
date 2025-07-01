import src.constants as constants
import torch
import torch.nn as nn
from transformers import AutoModel

class ColaClassifier(nn.Module):
    def __init__(self, model_name=constants.BASE_MODEL, num_classes=2):
        super(ColaClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0]
        return self.classifier(cls_output)
    
    def load_from_checkpoint(self, model_path, device="cpu"):
        """
        Load the model weights from a specified path.
        """
        self.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        self.model.to(device)
        self.eval()


# #import torch
# import torch.nn as nn
# import pytorch_lightning as pl
# import torch.nn.functional as F
# from transformers import AutoModel
# from sklearn.metrics import accuracy_score


# class ColaModel(pl.LightningModule):
#     def __init__(self, model_name="google/bert_uncased_L-2_H-128_A-2", lr=1e-2):
#         super(ColaModel, self).__init__()
#         self.save_hyperparameters()

#         self.bert = AutoModel.from_pretrained(model_name)
#         self.W = nn.Linear(self.bert.config.hidden_size, 2)
#         self.num_classes = 2

#     def forward(self, input_ids, attention_mask):
#         outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

#         h_cls = outputs.last_hidden_state[:, 0]
#         logits = self.W(h_cls)
#         return logits

#     def training_step(self, batch, batch_idx):
#         logits = self.forward(batch["input_ids"], batch["attention_mask"])
#         loss = F.cross_entropy(logits, batch["label"])
#         self.log("train_loss", loss, prog_bar=True)
#         return loss

#     def validation_step(self, batch, batch_idx):
#         logits = self.forward(batch["input_ids"], batch["attention_mask"])
#         loss = F.cross_entropy(logits, batch["label"])
#         _, preds = torch.max(logits, dim=1)
#         val_acc = accuracy_score(preds.cpu(), batch["label"].cpu())
#         val_acc = torch.tensor(val_acc)
#         self.log("val_loss", loss, prog_bar=True)
#         self.log("val_acc", val_acc, prog_bar=True)

#     def configure_optimizers(self):
#         return torch.optim.Adam(self.parameters(), lr=self.hparams["lr"])
