import pickle
import numpy as np
import os
import sys

# Add root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import settings

def load_model():
    """ Load the trained model from disk. """
    if not os.path.exists(settings.MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {settings.MODEL_PATH}. Please run training first.")
    
    with open(settings.MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    return model

def make_prediction(model, features):
    """
    Make a prediction given a model and features.
    features should be a 1D or 2D array-like.
    """
    if len(np.array(features).shape) == 1:
        features = [features]
    
    prediction = model.predict(features)
    return prediction[0]
