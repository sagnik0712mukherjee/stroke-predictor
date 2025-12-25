import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle
import sys
import os

# Add root to path to allow imports from config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import settings
from src.preprocessing import preprocess_df, get_features_targets

def train_model():
    """
    Load data, preprocess, train model and save to disk.
    """
    print(f"Loading data from {settings.DATA_PATH}...")
    df = pd.read_csv(settings.DATA_PATH)
    
    print("Preprocessing data...")
    df_processed = preprocess_df(df, settings.CATEGORICAL_COLUMNS)
    X, y = get_features_targets(df_processed, settings.DROP_COLUMNS, settings.TARGET_COLUMN)
    
    print("Training model...")
    classifier = RandomForestClassifier(**settings.MODEL_PARAMS)
    classifier.fit(X, y.values.ravel())
    
    print(f"Saving model to {settings.MODEL_PATH}...")
    with open(settings.MODEL_PATH, 'wb') as f:
        pickle.dump(classifier, f)
    
    print("Training complete.")

if __name__ == "__main__":
    train_model()
