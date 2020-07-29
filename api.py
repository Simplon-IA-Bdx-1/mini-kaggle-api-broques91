import numpy as np
import pandas as pd
from flask import Flask, request, render_template
from sklearn.metrics import roc_auc_score
import joblib

app = Flask(__name__)
df_test = pd.read_csv('data/test.csv')


@app.route('/submit', methods=["POST"])
def submit():

    predictions = request.files['file']
    df_pred = pd.read_csv(predictions)

    y_test = df_test['SeriousDlqin2yrs']
    y_pred = df_pred['SeriousDlqin2yrs']

    output = roc_auc_score(y_pred, y_test)

    return 'Score {}'.format(output)

if __name__ == "__main__":
    app.run(debug=True)