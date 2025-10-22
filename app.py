# app.py
from flask import Flask, render_template, request, redirect, url_for
import joblib
import numpy as np
import os

app = Flask(__name__)

# ---- Load saved objects ----
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'svm_model.joblib')
LE_PATH    = os.path.join(os.path.dirname(__file__), 'label_encoder.joblib')
SCALER_PATH = os.path.join(os.path.dirname(__file__), 'scaler.joblib')  # optional

model = joblib.load(MODEL_PATH)
label_encoder = joblib.load(LE_PATH)

# Load scaler if you used one for training; otherwise set to None
if os.path.exists(SCALER_PATH):
    scaler = joblib.load(SCALER_PATH)
else:
    scaler = None

# The list and ordering of feature names must match the order used for training
FEATURES = ['precipitation', 'temp_max', 'temp_min', 'wind', 'temp_avg', 'temp_diff']

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html', features=FEATURES)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Read inputs in the same order as FEATURES
        vals = []
        for f in FEATURES:
            v = request.form.get(f)
            if v is None or v.strip() == '':
                # if empty, you can choose a default or return an error
                return render_template('result.html', error=f"Missing value for {f}")
            # convert to float
            vals.append(float(v))

        X = np.array(vals).reshape(1, -1)

        # Apply scaler if used during training
        if scaler is not None:
            X = scaler.transform(X)

        pred_num = model.predict(X)  # numeric label(s)
        # If predict returns array, take first
        if hasattr(pred_num, '__iter__'):
            pred_num = pred_num[0]

        # Convert numeric prediction back to original string label
        pred_str = label_encoder.inverse_transform([pred_num])[0]

        return render_template('result.html', prediction=pred_str, inputs=dict(zip(FEATURES, vals)))

    except Exception as e:
        return render_template('result.html', error=str(e))

if __name__ == '__main__':
    # debug=False for production
    app.run(host='0.0.0.0', port=5000, debug=True)
