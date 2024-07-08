from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import os

app = Flask(__name__, template_folder='templates')

# Load the trained model pipeline
model_path = os.path.join(os.path.dirname(__file__), '../Predictive Models/car_price_prediction_random_forest_library.pkl')
pipeline_loaded = joblib.load(model_path)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    data = request.form.to_dict(flat=True)

    sample_data = {
        'make': [data['make']],
        'model': [data['model']],
        'year': [float(data['year'])],
        'engine': [float(data['engine'])],
        'transmission': [data['transmission']],
        'fuel': [data['fuel']],
        'mileage': [float(data['mileage'])]
    }

    sample_df = pd.DataFrame(sample_data)

    # Predict using the loaded pipeline
    predicted_price = pipeline_loaded.predict(sample_df)

    return jsonify({'price': predicted_price[0]})


if __name__ == '__main__':
    app.run(debug=True)
