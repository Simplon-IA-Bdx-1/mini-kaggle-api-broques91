from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import joblib

# Load Dataset
df_train = pd.read_csv('data/train.csv', sep=',', index_col=0)
df_test = pd.read_csv('data/test.csv', sep=',', index_col=0)
X_test = df_test.drop(['SeriousDlqin2yrs'], axis=1)

# Features / Target
x = df_train.drop('SeriousDlqin2yrs', axis=1)
y = df_train['SeriousDlqin2yrs']

# Train Test Split
X_train, X_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)

# Train Classifier
tree_clf = DecisionTreeClassifier()
tree_clf.fit(X_train,y_train)

# Predict
y_pred = tree_clf.predict(X_test)
df_test['SeriousDlqin2yrs'] = y_pred

# Export Predictions
df_test.to_csv('test2-predictions.csv')

# Save Classifier
joblib.dump(tree_clf, 'model/model.pkl')

