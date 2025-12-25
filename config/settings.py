import os

# Base directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# File paths
DATA_PATH = os.path.join(BASE_DIR, 'src/data/healthcare-dataset-stroke-data.csv')
MODEL_PATH = os.path.join(BASE_DIR, 'model.pkl')

# Data configuration
CATEGORICAL_COLUMNS = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
DROP_COLUMNS = ['id', 'stroke']
TARGET_COLUMN = 'stroke'

# Model configuration
MODEL_PARAMS = {
    'n_estimators': 200,
    'max_features': 'log2',
    'criterion': 'gini',
    'class_weight': 'balanced'
}
