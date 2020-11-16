"""
Run the prediction on Azure machine learning.
"""
import os
import json
import numpy as np
from tensorflow.keras.models import load_model


def init():
    """
    Load the model
    """
    global model
    # AZUREML_MODEL_DIR is an environment variable created during deployment.
    # It is the path to the model folder (./azureml-models/$MODEL_NAME/$VERSION)
    # For multiple models, it points to the folder containing all deployed models (./azureml-models)
    model_path = os.path.join(os.getenv("AZUREML_MODEL_DIR"), "keras_lenet.h5")
    model = load_model(model_path)


def run(raw_data):
    """
    Prediction
    """
    data = json.loads(raw_data)["data"]
    data = np.array(data)
    data = data.reshape((1, 28, 28, 1))
    # make prediction
    y_hat = model.predict(data)
    return float(np.argmax(y_hat))
