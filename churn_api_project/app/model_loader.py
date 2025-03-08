import joblib
import numpy as np

# Load the pre-trained model
model = joblib.load("churn_model.pkl")

def predict_churn(features):
    """
    Takes user input features as a list and returns churn probability.
    """
    input_data = np.array(features).reshape(1, -1)  # Reshape for model
    prediction = model.predict(input_data)[0][0]  # Extract probability from array
    return float(prediction)  # Convert NumPy float to Python float

print("Model input shape:", model.input_shape)