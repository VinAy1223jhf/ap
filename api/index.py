from flask import Flask, request, jsonify
import pickle
from sklearn.preprocessing import StandardScaler
from flask_cors import CORS

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the models
scaler = StandardScaler()
diabetes_model = pickle.load(open('diabetes_model.sav', 'rb'))
heart_model = pickle.load(open('heart_model.sav', 'rb'))
parkinson_model = pickle.load(open('parkinson_model.sav', 'rb'))

# Diabetes prediction route
@app.route('/predict/diabetes', methods=['POST'])
def predict_diabetes():
    try:
        data = request.json
        required_fields = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                           'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

        if not all(field in data for field in required_fields):
            return jsonify({'error': 'Missing fields in request'}), 400

        features = [float(data[field]) for field in required_fields]
        prediction = diabetes_model.predict([features])

        return jsonify(result=int(prediction[0]))

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': 'An error occurred during prediction'}), 500

# Heart disease prediction route
@app.route('/predict/heart', methods=['POST'])
def predict_heart():
    try:
        data = request.json
        required_fields = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
                           'restecg', 'thalach', 'exang', 'oldpeak', 'slope',
                           'ca', 'thal']

        if not all(field in data for field in required_fields):
            return jsonify({'error': 'Missing fields in request'}), 400

        features = [float(data[field]) for field in required_fields]
        prediction = heart_model.predict([features])

        return jsonify({'prediction': int(prediction[0])})

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': 'An error occurred during prediction'}), 500

# Parkinson's prediction route
@app.route('/predict/parkinsons', methods=['POST'])
def predict_parkinsons():
    try:
        data = request.json
        required_fields = ['fo', 'fhi', 'flo', 'Jitter_percent', 'Jitter_Abs',
                           'RAP', 'PPQ', 'DDP', 'Shimmer', 'Shimmer_dB',
                           'APQ3', 'APQ5', 'APQ', 'DDA', 'NHR', 'HNR', 'RPDE',
                           'DFA', 'spread1', 'spread2', 'D2', 'PPE']

        if not all(field in data for field in required_fields):
            return jsonify({'error': 'Missing fields in request'}), 400

        features = [float(data[field]) for field in required_fields]
        prediction = parkinson_model.predict([features])

        return jsonify({'prediction': int(prediction[0])})

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': 'An error occurred during prediction'}), 500

# Vercel handler function
def handler(event, context):
    return app(event, context)
