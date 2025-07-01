import constants

from inference import ColaInference

import logging
logging.basicConfig(level=logging.INFO)

def main():
    model_path = f"{constants.CHECKPOINT_DIR}/best_model.pt"
    inference_engine = ColaInference(model_path, model_name=constants.BASE_MODEL)

    logging.info("Enter a sentence to classify as grammatical/ungrammatical ('exit' to quit):")

    while True:
        sentence = input(">> ")
        if sentence.lower() == "exit":
            break

        pred, conf = inference_engine.predict(sentence)
        label = constants.GRAMMATICAL if pred == 1 else constants.UMGRAMMATICAL
        print(f"Prediction: {label} (confidence: {conf:.2f})")

if __name__ == "__main__":
    main()
