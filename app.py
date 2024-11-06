# app.py
from flask import Flask, request, jsonify , render_template
import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)

# Load your trained model (assuming you have saved it as model.pkl)
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template('diabetes.html')
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    glucose = data['glucose']
    bmi = data['bmi']
    insulin = data['insulin']

    # Prepare the input data for prediction
    input_features = np.array([[glucose, bmi, insulin]])

    # Predict using the loaded model
    prediction = model.predict(input_features)

    # Format the result
    result = 'Diabetic' if prediction[0] == 1 else 'Non-diabetic'
    return jsonify({'prediction': result})


if __name__ == '__main__':
    app.run(debug=True)
