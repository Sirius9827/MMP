import argparse
import numpy as np  
from rdkit import Chem  
from rdkit.Chem import AllChem 
from sklearn.model_selection import train_test_split
#from sklearn.svm import SVC
from xgboost import XGBClassifier 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error, mean_absolute_error, f1_score

import pandas as pd  
import pandas_flavor as pf  
from descriptors.rdDescriptors import RDKit2D
#from descriptors.rdNormalizedDescriptors import RDKit2DNormalized

from descriptastorus.descriptors.rdNormalizedDescriptors import RDKit2DNormalized
from sklearn.impute import SimpleImputer 
#import tdc required libraries
from tdc.utils import retrieve_label_name_list
from tdc.single_pred import Tox
from tdc.single_pred import HTS
from tdc.single_pred import QM
from tdc.single_pred import ADME

from rdkit.DataStructs import cDataStructs
from DeepPurpose import CompoundPred as models
from DeepPurpose.utils import *

# from sklearn.svm import SVC  # Uncomment if SVM is used

# Define command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Cheminformatics Model Training and Evaluation")
    #parser.add_argument("--data_path", type=str, required=True, help="Path to the dataset directory")
    parser.add_argument("--dataset_name", type=str, required=True, help="Name of the dataset file")
    parser.add_argument("--model", type=str, required=True, choices=['SVM', 'XGB', 'RF'], help="Model to use")
    #parser.add_argument("--n_jobs", type=int, default=32, help="Number of jobs to run in parallel")
    return parser.parse_args()

def process_dataset(dataset_name):
    # Convert dataset_name to the expected case, e.g., uppercase
    #dataset_name = dataset_name.upper()  # Ensure dataset_name is uppercase

    # Retrieve label names based on the dataset
    label_lists = retrieve_label_name_list(dataset_name)
    
    # Determine the dataset type based on its name and initialize accordingly
    if dataset_name in ['ClinTox', 'ToxCast', 'Tox21']:
        # Toxicity prediction datasets
        data = Tox(name = dataset_name, label_name = label_lists[0])
    elif dataset_name in ['SARSCoV2_Vitro_Touret', 'HIV']:
        # HTS datasets
        data = HTS(name = dataset_name)
    elif dataset_name in ['QM7b', 'QM8', 'QM9']:
        # QM datasets
        data = QM(name = dataset_name, label_name = label_lists[0])
    elif dataset_name in ['ADME']:
        data = ADME(name = dataset_name).get_data(format = 'dict')
        X, y = data['Drug'], data['Y']
    
    else:
        raise ValueError(f"Dataset {dataset_name} is not recognized.")
    
    # Get the split for the dataset
    split = data.get_split(method = 'scaffold')
    # Get the SMILES and labels for the dataset
    X_train, y_train = split['train']['Drug'], split['train']['Y']
    X_val, y_val = split['valid']['Drug'], split['valid']['Y']
    # print(X_train)
    # print(X_val)
    # print(type(X_train))


    #Create dataset with RDKit2D descriptors
    train_rdkit2d = RDKit2DNormalized()
    X_train = train_rdkit2d.processSmiles(X_train)
    val_rdkit2d = RDKit2DNormalized()
    X_val = val_rdkit2d.processSmiles(X_val)

    sequences = X_train
    # Calculate the lengths of all sequences
    lengths = [len(seq) for seq in sequences]

    # Find the unique lengths to identify inhomogeneity
    unique_lengths = set(lengths)

    if len(unique_lengths) > 1:
        print("Inhomogeneous sequence lengths found:", unique_lengths)
        # Optional: Print the sequences with unexpected lengths for inspection
        for i, seq in enumerate(sequences):
            if len(seq) not in unique_lengths:
                print(f"Sequence at index {i} has unexpected length {len(seq)}: {seq}")
    else:
        print("All sequences have a homogeneous length.")
    
    return X_train, X_val, y_train, y_val

# # Main function
# if __name__ == "__main__":
#     args = parse_args()
#     X_train, X_val, y_train, y_val = process_dataset(args.dataset_name)
#     # model = get_model(args.model)
#     # train_evaluate_model(X_train, X_test, y_train, y_test, model)
#     print(X_val)

# Preprocess dataset
# def preprocess_dataset(data_split, dataset_name):
#     data_split = dataset(dataset_name)
#     # Assuming 'Drug' column for SMILES and 'Y' for labels/targets
#     # X = dataset.drop('Y', axis=1)
#     # y = dataset['Y']
#     data_split = data_process(X_drug = data_split.Drug.values, y = data_split.Y.values)
#     imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
#     X_imputed = imputer.fit_transform(X)
#     return train_test_split(X_imputed, y, test_size=0.2, random_state=42)

# Define model based on user input
def get_model(model_name):
    if model_name == 'XGB':
        return XGBClassifier()
    elif model_name == 'RF':
        return RandomForestClassifier()
    # elif model_name == 'SVM':
    #     return SVC()
    else:
        raise ValueError("Model not supported")

# Train and evaluate model
def train_evaluate_model(X_train, X_val, y_train, y_val, model):
    model.fit(X_train, y_train)
    predictions = model.predict(X_val)
    accuracy = accuracy_score(y_val, predictions)
    auc = roc_auc_score(y_val, predictions)
    # Add more metrics as needed
    print(f"Accuracy: {accuracy}, AUC: {auc}")

# Main function
if __name__ == "__main__":
    args = parse_args()
    X_train, X_val, y_train, y_val = process_dataset(args.dataset_name)
    model = get_model(args.model)
    train_evaluate_model(X_train, X_val, y_train, y_val, model)

exit(0)
def preprocess_dataset(args):

    # train_val = "data/admet_group/ames/train_val.csv"
    # test = "data/admet_group/ames/test.csv"
    # train_val = pd.read_csv(train_val)
    # test = pd.read_csv(test)
    train = pd.read_csv(f"{args.data_path}/{args.dataset}/{args.dataset}.csv")
    cache_file_path = f"{args.data_path}/{args.dataset}/{args.dataset}_{args.path_length}.pkl"    

    train_val_smiles = train_val["Drug"].tolist()
    test_smiles = test["Drug"].tolist()

# Create dataset with RDKit2D descriptors
train_val_rdkit2d = RDKit2D()
t_v_mols, train_val_descriptors = train_val_rdkit2d.processSmiles(train_val_smiles)
X = train_val_descriptors
y = train_val["Y"]

test_rdkit2d = RDKit2D()
ts_mols, test_descriptors = test_rdkit2d.processSmiles(test_smiles)
X_test = test_descriptors
y_test = test["Y"]

# Impute missing values (NaN) with the mean of the corresponding feature  
imputer = SimpleImputer(missing_values=float('nan'), strategy='mean')  
X = imputer.fit_transform(X)  
X_test = imputer.transform(X_test)  

predictions_list = []

#for seed in [1, 2, 3, 4, 5]:
    
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=1234)
predictions = {}

#params
# params = {
#     'booster': 'gbtree',
#     'objective': 'reg:gamma',
#     'gamma': 0.1,
#     'max_depth': 5,
#     'lambda': 3,
#     'subsample': 0.7,
#     'colsample_bytree': 0.7,
#     'min_child_weight': 3,
#     'slient': 1,
#     'eta': 0.1,
#     'seed': 1000,
#     'nthread': 4,
# }


# Train XGBoost Model  
xgb_model = XGBClassifier()  # You can adjust the parameters  
xgb_model.fit(X_train, y_train)  
y_pred_test_xgb = xgb_model.predict(X_test)  
accuracy_xgb = accuracy_score(y_test, y_pred_test_xgb)
auc_xgb = roc_auc_score(y_test, y_pred_test_xgb)

print("XGBoost Accuracy:", accuracy_xgb)  
print("AUC", auc_xgb)

print("Accuracy:", accuracy_xgb)  
print("Classification Report:")  


    
predictions[name] = y_pred_test_xgb
predictions_list.append(predictions)

#results = group.evaluate_many(predictions_list)