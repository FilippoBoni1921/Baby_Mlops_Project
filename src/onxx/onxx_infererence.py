import numpy as np
import onnxruntime as ort
from scipy.special import softmax

from src.data import DatasetLoader
from src.utils import timing

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ColaONNXPredictor:
    def __init__(self, model_path):
        self.ort_session = ort.InferenceSession(model_path)
        self.processor = DatasetLoader()
        self.lables = ["unacceptable", "acceptable"]

    @timing
    def predict(self, text):
        inference_sample = {"sentence": text}
        processed = self.processor.tokenize_data(inference_sample)

        ort_inputs = {
            "input_ids": np.atleast_2d(processed["input_ids"]),
            "attention_mask": np.atleast_2d(processed["attention_mask"]),
        }
        ort_outs = self.ort_session.run(None, ort_inputs)
        scores = softmax(ort_outs[0])[0]
        predictions = []
        for score, label in zip(scores, self.lables):
            predictions.append({"label": label, "score": float(score)})
        
        return predictions


if __name__ == "__main__":
    sentence = "The boy is sitting on a bench"
    predictor = ColaONNXPredictor("./models/model.onnx")
    prediction = predictor.predict(sentence)
    logger.info("Prediction: %s", prediction)
    sentences = ["The boy is sitting on a bench"] * 10
    for sentence in sentences:
        predictor.predict(sentence)