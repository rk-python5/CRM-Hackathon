import requests
import json

BASE_URL = "http://127.0.0.1:5000"  # Change if deployed elsewhere

def test_api():
    test_data = {
        "features": [30, 1, 0, 70.35, 1, 0, 2, 0, 1, 1, 0, 1, 0, 2, 1, 1, 2, 1]  # Replace with actual feature values
    }

    response = requests.post(f"{BASE_URL}/predict", json=test_data)

    if response.status_code == 200:
        print("Test Passed ✅")
        print("Response:", response.json())
    else:
        print("Test Failed ❌")
        print("Error:", response.text)

if __name__ == "__main__":
    test_api()