import pickle
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load the updated model
model = load_model("customer_churn_model-3.h5")

# Load the updated scaler
with open("scaler-3.pkl", "rb") as f:
    scaler = pickle.load(f)

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Churn Prediction API is running!"})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json  # Expecting JSON input
        features = np.array(data["features"]).reshape(1, -1)

        # Ensure correct input shape
        if features.shape[1] != 30:
            return jsonify({"error": f"Expected 30 features, but got {features.shape[1]}"}), 400

        # Preprocess input data
        scaled_features = scaler.transform(features)

        # Make a prediction
        prediction = model.predict(scaled_features)
        predicted_class = int(prediction[0][0] > 0.5)  # Convert to binary (0 or 1)

        return jsonify({"churn_probability": float(prediction[0][0]), "churn_prediction": predicted_class})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)