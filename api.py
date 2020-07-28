import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
from sklearn.metrics import accuracy_score
import joblib

app = Flask(__name__)
model = joblib.load(open('model/model.pkl', 'rb'))
df_test = pd.read_csv('data/test.csv')


@app.route('/submit', methods=["POST"])
def submit():

    predictions = request.files['file']
    df_pred = pd.read_csv(predictions)

    y_test = df_test['SeriousDlqin2yrs']
    y_pred = df_pred['SeriousDlqin2yrs']

    output = accuracy_score(y_pred, y_test)

    return 'Score {}'.format(output)

if __name__ == "__main__":
    app.run(debug=True)