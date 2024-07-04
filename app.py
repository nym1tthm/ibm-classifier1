import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract features from the form
        int_features = [float(x) for x in request.form.values()]
        final_features = [np.array(int_features)]
        
        # Predict using the model
        prediction = model.predict(final_features)
        output = round(prediction[0], 2)
        
        # Return the prediction in the HTML response
        return render_template('index.html', prediction_text='Potability is {}'.format(output))
    except Exception as e:
        return render_template('index.html', prediction_text='Error: {}'.format(e))

@app.route('/predict_api', methods=['POST'])
def predict_api():
    try:
        # Extract JSON data
        data = request.get_json(force=True)
        features = [float(x) for x in data.values()]
        final_features = [np.array(features)]
        
        # Predict using the model
        prediction = model.predict(final_features)
        output = prediction[0].item()  # Convert to native Python type
        
        # Return the prediction as JSON
        return jsonify(output)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
