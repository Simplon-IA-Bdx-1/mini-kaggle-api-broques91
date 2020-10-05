from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import joblib

# Load Dataset
df_train = pd.read_csv('data/train2.csv', sep=',')
df_test = pd.read_csv('data/test2.csv', sep=',')

# ID
for df in [df_train, df_test]:
    df.rename(columns={'Unnamed: 0':'ID'}, inplace=True)

# Features / Target
X_train = df_train.drop('SeriousDlqin2yrs', axis=1)
y_train = df_train['SeriousDlqin2yrs']
X_test = df_test.drop(['SeriousDlqin2yrs'], axis=1)
y_test = df_test['SeriousDlqin2yrs']

# Train Classifier
model = DecisionTreeClassifier()
model.fit(X_train,y_train)

# Predict
y_pred = model.predict_proba(X_test)[:,1]

# DataFrame Predictions
data = {'ID': X_test['ID'], 'SeriousDlqin2yrs': y_pred}
predictions = pd.DataFrame(data=data)

# Save Predictions
predictions.to_csv('test2-predictions.csv')

# Save Classifier
joblib.dump(model, 'model/model.pkl')
