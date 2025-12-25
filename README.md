# Stroke Prediction System

A production-grade machine learning application for predicting the likelihood of a stroke based on patient data using a Random Forest Classifier.

## Project Structure

```text
.
├── config/
│   ├── __init__.py
│   └── settings.py           # Configuration and hyperparameters
├── src/
│   ├── __init__.py
│   ├── preprocessing.py      # Data cleaning and encoding
│   ├── train.py              # Model training logic
│   └── predict.py            # Inference logic
├── templates/
│   └── index.html            # Web interface
├── main.py                   # Flask application entry point
├── requirements.txt          # Python dependencies
├── .gitignore                # Git ignore rules
└── healthcare-dataset-stroke-data.csv # Dataset
```

## Features

- **Modular Architecture**: Clean separation of concerns between preprocessing, training, and inference.
- **Configurable**: Hyperparameters and file paths are centralized in `config/settings.py`.
- **Production Ready**: Includes error handling, structured logging (planned), and a clean Flask interface.
- **Data Preprocessing**: Automated handling of categorical encoding and missing value imputation.

## Setup and Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd stroke-predictor
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Train the model**:
   ```bash
   python src/train.py
   ```
   This will generate `model.pkl` in the root directory.

4. **Run the application**:
   ```bash
   python main.py
   ```
   The app will be available at `http://127.0.0.1:5000/`.

## Usage

Fill in the patient details in the web form and click "Predict" to see the stroke probability.

## Dataset

The model uses the [Stroke Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset) from Kaggle.
