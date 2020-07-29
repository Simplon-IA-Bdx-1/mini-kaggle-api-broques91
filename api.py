import numpy as np
import pandas as pd
from flask import Flask, request, render_template
from sklearn.metrics import roc_auc_score
import joblib


# Start Flask
app = Flask(__name__)

# Test Data
df_test = pd.read_csv('data/test2.csv')

# Default webpage
@app.route('/')
def home():
    return render_template('index.html')

# Submit
@app.route('/submit', methods=["POST"])
def submit():

    # Load requested file
    predictions = request.files['file']
    df_pred = pd.read_csv(predictions)

    # Prediction / True values
    y_test = df_test['SeriousDlqin2yrs']
    y_pred = df_pred['SeriousDlqin2yrs']

    # Compute score
    output = roc_auc_score(y_pred, y_test)

    return 'Score {}'.format(output)

if __name__ == "__main__":
    app.run(debug=True)