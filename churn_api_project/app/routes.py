from flask import Blueprint, request, jsonify
from app.model_loader import predict_churn

api_bp = Blueprint("api", __name__)

@api_bp.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        features = data["features"]  # Expecting a list of feature values
        prediction = predict_churn(features)
        return jsonify({"churn_probability": prediction})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 400