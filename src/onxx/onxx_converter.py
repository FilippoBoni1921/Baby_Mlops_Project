import torch
import hydra
import logging

from omegaconf.omegaconf import OmegaConf
import configs.configs_py.main_config

import src.constants as constants
from src.model import ColaClassifier
from src.data import DatasetLoader

logger = logging.getLogger(__name__)

@hydra.main(config_name="config",version_base=None)
def convert_model(cfg):
    root_dir = hydra.utils.get_original_cwd()

    model_path = f"{constants.CHECKPOINT_DIR}/best_model.pt"
    logger.info(f"Loading pre-trained model from: {model_path}")

    model = ColaClassifier()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    data_model = DatasetLoader(cfg.model.tokenizer, cfg.processing.batch_size, cfg.processing.max_length)

    data_model.prepare()
    train_loader, _ = data_model.get_dataloaders()
    input_batch = next(iter(train_loader))
    input_sample = {
        "input_ids": input_batch["input_ids"][0].unsqueeze(0),
        "attention_mask": input_batch["attention_mask"][0].unsqueeze(0),
    }
    
    # Export the model
    logger.info(f"Converting the model into ONNX format")
    torch.onnx.export(
        model,  # model being run
        (
            input_sample["input_ids"],
            input_sample["attention_mask"],
        ),  # model input (or a tuple for multiple inputs)
        f"{root_dir}/models/model.onnx",  # where to save the model (can be a file or file-like object)
        export_params=True,
        opset_version=14,
        input_names=["input_ids", "attention_mask"],  # the model's input names
        output_names=["output"],  # the model's output names
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "output": {0: "batch_size"},
        },
    )

    logger.info(
        f"Model converted successfully. ONNX format model is at: {root_dir}/models/model.onnx"
    )


if __name__ == "__main__":
    convert_model()
