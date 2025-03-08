from flask import Flask
from app.routes import api_bp  # Import API routes

def create_app():
    app = Flask(__name__)
    app.register_blueprint(api_bp)  # Register routes
    return app