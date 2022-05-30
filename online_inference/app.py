import logging
import os

import joblib
import uvicorn
from fastapi import FastAPI

from src.models import ModelInput, ModelOutput, make_prediction
from src.s3_loader import s3_loader

MODEL_FOLDER = 'models'
MODEL_FILE = 'estimator.pkl'
HOST = "0.0.0.0"
PORT = 5000

app = FastAPI(title="Heart Disease Model API",
              description="A simple model to make a Heart Disease Condition",
              version="0.0.1")
model = None


@app.on_event('startup')
def load_model():
    """Load model for further prediction"""
    instance_logger_object = logging.getLogger(__name__)
    instance_logger_object.info('Loading model...')
    model_path = os.path.join(os.getcwd(), MODEL_FOLDER, MODEL_FILE)
    if model_path is None:
        error = f"Model is not provided: model_path is absent:\n{model_path}"
        instance_logger_object.error(error)
        raise RuntimeError(error)
    global model
    model = joblib.load(model_path)
    instance_logger_object.info('Model is successfully loaded')

@app.on_event('load_from_s3')
def load_from_s3():
    """Load model from s3"""
    instance_logger_object = logging.getLogger(__name__)
    instance_logger_object.info('Loading model...')
    s3_loader()
    global model
    model = joblib.load(model_path)
    instance_logger_object.info('Model is successfully loaded')


@app.get('/')
async def root():
    """Root app message"""
    return {
        'message':
        "MADE ML IN PROD Homework 2: "
        "Heart disease predictor inference"
    }


@app.get('/health')
async def health_check():
    """Health checkup function"""
    instance_logger_object = logging.getLogger(__name__)
    if model:
        instance_logger_object.info("The model is ready for inference")
        return 200
    else:
        instance_logger_object.info("Health checkup failed")
        return 204


@app.get('/predict', response_model=list)
async def predict(request: ModelInput):
    """Predict model inference"""
    instance_logger_object = logging.getLogger(__name__)
    instance_logger_object.info("Making predictions...")
    prediction = make_prediction(request.data, request.feature_names, model)
    instance_logger_object.info("Prediction successfully made")
    return prediction


if __name__ == "__main__":
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    uvicorn.run(app,
                host=HOST,
                port=os.getenv("PORT", PORT)
                )
