from fastapi import FastAPI
from pathlib import Path
from src.onxx.onxx_infererence import ColaONNXPredictor
app = FastAPI(title="MLOps Basics App")

BASE_DIR = Path(__file__).resolve().parent  # /app/
MODEL_PATH = BASE_DIR.parent / "models" / "model.onnx"

predictor = ColaONNXPredictor(str(MODEL_PATH))


@app.get("/")
async def homr_page():
    return "<h2>Sample prediction API</h2>"

@app.get("/predict")
async def get_prediction(text: str):
    result = predictor.predict(text)
    return result

#Now Let's run the application using the command
#uvicorn app:app --ip 0.0.0.0 --port 8000 --reload
#uvicorn app.app:app --host 0.0.0.0 --port 8000 --reload
#uvicorn	Starts the ASGI web server to serve your FastAPI app
#app:app	module_name:app_instance →app.py is the filename, and inside it there’s an app = FastAPI() instance . app.app because I wanna run it from main working directory
#--host 0.0.0.0	Makes the app accessible on any network interface (not just localhost). Needed inside Docker or remote VMs.
#--port 8000	Binds the server to port 8000. You can change it to any free port.
#--reload	Enables auto-reloading during development: the server restarts when you modify code. 🔁 (Don’t use in production.)