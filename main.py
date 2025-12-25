from flask import Flask, request, render_template
from src.predict import load_model, make_prediction

app = Flask(__name__)

# Load model at startup
try:
    model = load_model()
except FileNotFoundError:
    print("Warning: model.pkl not found. Please run 'python src/train.py' to generate it.")
    model = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return render_template('index.html', prediction_text="Error: Model not loaded. Train the model first.")
        
    try:
        # Extract features from form
        int_features = [int(x) for x in request.form.values()]
        # Make prediction
        output = make_prediction(model, int_features)
        
        return render_template('index.html', prediction_text=f'The chances of patient having a stroke is {output}')
    except Exception as e:
        return render_template('index.html', prediction_text=f"Error in prediction: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
