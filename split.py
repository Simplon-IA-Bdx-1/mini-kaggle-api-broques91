import pandas as pd
import os 

# Create Data Directory
path = "./data"

try:
    os.makedirs(path)
    print ("Successfully created the directory %s" % path)
except FileExistsError: 
    print("Directory " , path ,  " already exists")  
else:
    print ("Creation of the directory %s failed" % path)

# Load Dataset
df = pd.read_csv('https://oml-data.s3.amazonaws.com/kaggle-give-me-credit-train.csv', index_col=0)
df = df.dropna()
# print(df.columns)

# Split Dataset
train_set = df.sample(frac=0.8, random_state=42)
test_set = df.drop(train_set.index)

print("######## Split complete #########")
print(f"Train set : {len(train_set)} entries")
print(f"Test set : {len(test_set)} entries")

# Save New Datasets
train_set.to_csv('data/train2.csv')
test_set.to_csv('data/test2.csv')


