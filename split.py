import pandas as pd


# Load Dataset
df = pd.read_csv('data/cs-training.csv', sep=',', index_col=0)
df = df.dropna()
print(df.columns)

# Split Dataset
train_set = df.sample(frac=0.8, random_state=42)
test_set = df.drop(train_set.index)
test_set2 = test_set.copy()
test_set2['SeriousDlqin2yrs'] = ''

# Save New Datasets
train_set.to_csv('data/train.csv')
test_set.to_csv('data/test.csv')
test_set2.to_csv('data/test2.csv')


