from sys import argv
import pandas as pd
from sklearn.model_selection import train_test_split
from argparse import ArgumentParser
import os
parser = ArgumentParser(description='dataset splitting')
parser.add_argument('-d', '--dataset', type=str, help='dataet name. select from esol, freesolv, lipop')
directory_name = "datasets/"

args = parser.parse_args()
file_name = args.dataset
csv_file = args.dataset + ".csv"
args = parser.parse_args()
# Load the dataset
data = pd.read_csv(os.path.join(directory_name, file_name, csv_file))

# Shuffle the dataset
data = data.sample(frac=1).reset_index(drop=True)

# Split the dataset into training (80%), validation (10%), and test (10%)
train, temp = train_test_split(data, test_size=0.2, random_state=42)
valid, test = train_test_split(temp, test_size=0.5, random_state=42)

# Save the datasets to new CSV files
train.to_csv(os.path.join(directory_name, file_name,'train.csv'), index=False)
valid.to_csv(os.path.join(directory_name, file_name, 'valid.csv'), index=False)
test.to_csv(os.path.join(directory_name, file_name, 'test.csv'), index=False)