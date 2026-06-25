from flask import Flask, request, jsonify
import joblib
import numpy as np

model = joblib.load('loan_approval_pipeline.pkl')

app = Flask(__name__)

@app.route('/')
def home():
    return ''' 
    <h2>Loan Approval Prediction</h2>
    '''

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json

    features = np.array([[
        
        data['no_of_dependents'],
        data['education'],
        data['self_employed'],
        data['income_annum'],
        data['loan_amount'],
        data['loan_term'],
        data['cibil_score'],
        data['residential_assets_value'],
        data['commercial_assets_value'],
        data['luxury_assets_value'],
        data['bank_asset_value'],


    ]])

    prediction = model.predict(features)

    probability = model.predict_proba(features)

    result = "Loan Approved"

    if prediction[0] == 0:
        result = "Loan Not Approved"

        confidence = np.max(probability) * 100

        return jsonify({
            'prediction': result,
            'confidence': str(confidence)
        })

if __name__ == '__main__':
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True
    )