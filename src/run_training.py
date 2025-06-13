import logging
import hydra

import configs.configs_py.register_config
from configs.configs_py.main_config import Config

import src.constants as constants
from src.data import DatasetLoader
from src.model import ColaClassifier
from src.train import Trainer

from omegaconf.omegaconf import OmegaConf

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

@hydra.main(config_path="../configs", config_name="config")
def main(cfg: Config):
    logger.info(OmegaConf.to_yaml(cfg, resolve=True))
    logger.info(f"Using the model: {cfg.model.name}")
    logger.info(f"Using the tokenizer: {cfg.model.tokenizer}")

    data_loader = DatasetLoader(
        cfg.model.tokenizer, cfg.processing.batch_size, cfg.processing.max_length
    )

    data_loader.prepare()
    train_loader, val_loader = data_loader.get_dataloaders()

    model = ColaClassifier(cfg.model.name)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        lr=cfg.training.lr1e-3,
        epochs=cfg.training.max_epochs,
        wandb_project=constants.WAND_PROJECT,
        wandb_entity=constants.WANDB_ENTITY,
        run_name=constants.WAND_RUN_NAME,
        early_stopping_patience=2,
        checkpoint_dir=constants.CHECKPOINT_DIR
    )

    trainer.train()

if __name__ == "__main__":
    main()
