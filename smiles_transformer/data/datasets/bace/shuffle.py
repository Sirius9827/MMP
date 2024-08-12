import pandas as pd
from sklearn.model_selection import train_test_split

# Load the dataset
data = pd.read_csv('bace.csv')

# Shuffle the dataset
data = data.sample(frac=1).reset_index(drop=True)

# Split the dataset into training (80%), validation (10%), and test (10%)
train, temp = train_test_split(data, test_size=0.2, random_state=42)
valid, test = train_test_split(temp, test_size=0.5, random_state=42)

# Save the datasets to new CSV files
train.to_csv('train.csv', index=False)
valid.to_csv('valid.csv', index=False)
test.to_csv('test.csv', index=False)