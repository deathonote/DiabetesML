
import sys
sys.path.append("/Users/kush/Documents/BITS /4-1/ML/A4/Flask/lib")
import os
import pickle
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd

# Define paths
base_dir = "/Users/kush/PycharmProjects/flaskProject"
nb_model_path = os.path.join(base_dir, 'naive_bayes_model.pkl')
perceptron_model_path = os.path.join(base_dir, 'perceptron_model.pkl')
scaler_path = os.path.join(base_dir, 'scaler.pkl')

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load the Naive Bayes model, Perceptron model, and scaler
with open(nb_model_path, 'rb') as nb_file:
    naive_bayes_model = pickle.load(nb_file)
with open(perceptron_model_path, 'rb') as p_file:
    perceptron_model = pickle.load(p_file)
with open(scaler_path, 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    # Extract and validate input features
    try:
        model_type = data["model_type"]
        features = {
            "Age": float(data["age"]),
            "Glucose": float(data["blood_sugar_level"]),
            "Insulin": float(data["insulin_level"]),
            "BMI": float(data["BMI"])
        }

        features_list = [features["Age"], features["Glucose"], features["Insulin"], features["BMI"]]
        features_df = pd.DataFrame([features_list], columns=["Age", "Glucose", "Insulin", "BMI"])

        # Scale the input features
        features_scaled = scaler.transform(features_df)

    except KeyError as e:
        return jsonify({"error": f"Missing parameter: {e.args[0]}"}), 400
    except ValueError:
        return jsonify({"error": "Invalid input: Ensure all features are numeric."}), 400

    # Predict using the selected model
    if model_type == "naive_bayes":
        prediction = naive_bayes_model.predict(features_scaled)
    elif model_type == "perceptron":
        prediction = perceptron_model.predict(features_scaled)
    else:
        return jsonify(
            {"error": "Invalid model type. Choose 'naive_bayes', 'perceptron', or 'custom_perceptron'."}), 400

    return jsonify({"prediction": int(prediction[0])})


if __name__ == '__main__':
    app.run(debug=True)