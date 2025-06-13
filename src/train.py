import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import os
import pandas as pd
import wandb
import src.constants as constants


import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class Trainer:
    def __init__(self, model, train_loader, val_loader, lr=1e-2, epochs=3, device=None,
                 wandb_project=constants.WAND_PROJECT, wandb_entity=None, run_name=None,
                 early_stopping_patience=3, checkpoint_dir="./checkpoints"):
        
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epochs = epochs
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()

        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Initialize a new W&B run for experiment tracking.
        # This creates a new run under the specified project and entity (W&B team or user),
        # and gives the run a human-readable name for easier identification in the W&B dashboard.
        #wandb.init(project=wandb_project, entity=wandb_entity, name=run_name)

        # Start tracking gradients and model parameters with W&B.
        # `log="all"` ensures both weights and gradients are logged during training,
        # allowing detailed visualizations like parameter histograms in the W&B UI.
        #wandb.watch(self.model, log="all")

        # Early stopping
        self.best_val_loss = float("inf")
        self.early_stopping_counter = 0
        self.patience = early_stopping_patience

    def train(self):
        for epoch in range(self.epochs):
            self.model.train()
            running_loss = 0.0

            for batch in tqdm(self.train_loader, desc=f"Epoch {epoch + 1}"):
                input_ids = batch[constants.INPUT_IDS].to(self.device)
                attention_mask = batch[constants.ATTENTION_MASK].to(self.device)
                labels = batch[constants.LABEL].to(self.device)

                logits = self.model(input_ids, attention_mask)
                loss = self.criterion(logits, labels)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

            avg_loss = running_loss / len(self.train_loader)
            logging.info(f"Epoch {epoch + 1} - Training Loss: {avg_loss:.4f}")
            #wandb.log({"train/loss": avg_loss, "epoch": epoch + 1})

            val_loss, _ = self.evaluate(epoch)

            # Early stopping logic
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.early_stopping_counter = 0
                # Save model checkpoint
                torch.save(self.model.state_dict(), os.path.join(self.checkpoint_dir, "best_model.pt"))
            else:
                self.early_stopping_counter += 1
                if self.early_stopping_counter >= self.patience:
                    logging.info("Early stopping triggered.")
                    break

    def evaluate(self, epoch):
        self.model.eval()
        predictions, true_labels, sentences = [], [], []
        total_loss = 0

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Evaluating"):
                input_ids = batch[constants.INPUT_IDS].to(self.device)
                attention_mask = batch[constants.ATTENTION_MASK].to(self.device)
                labels = batch[constants.LABEL].to(self.device)
                batch_sentences = batch.get(constants.SENTENCE, [""] * len(labels))  # optional

                logits = self.model(input_ids, attention_mask)
                loss = self.criterion(logits, labels)
                total_loss += loss.item()

                preds = torch.argmax(logits, dim=1)
                predictions.extend(preds.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())
                sentences.extend(batch_sentences)

        acc = accuracy_score(true_labels, predictions)
        avg_val_loss = total_loss / len(self.val_loader)
        logging.info(f"Validation Loss: {avg_val_loss:.4f}, Accuracy: {acc:.4f}")
        #wandb.log({"valid/loss": avg_val_loss, "valid/accuracy": acc, "epoch": epoch + 1})

        # Log incorrect predictions
        wrong_df = pd.DataFrame({
            constants.SENTENCE: sentences,
            constants.LABEL: true_labels,
            constants.PREDICTED: predictions
        })
        wrong_df = wrong_df[wrong_df[constants.LABEL] != wrong_df[constants.PREDICTED]]
        #wandb.log({
        #    constants.WRONG_PREDICTIONS: wandb.Table(dataframe=wrong_df)
        #})

        return avg_val_loss, acc
