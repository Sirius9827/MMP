import argparse
import numpy as np  
from rdkit import Chem  
from rdkit.Chem import AllChem 
from sklearn.model_selection import train_test_split
#from sklearn.svm import SVC
from sklearn.svm import SVC, SVR
from xgboost import XGBClassifier, XGBRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
# evaluation metrics for classification and regression
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error, mean_absolute_error, f1_score, r2_score, classification_report

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
from tdc.single_pred import ADME
# TDC benchmark group: admet_group
from tdc.benchmark_group import admet_group

from rdkit.DataStructs import cDataStructs
from DeepPurpose import CompoundPred as models
from DeepPurpose.utils import *

from allfingerprints import FingerprintProcessor

import csv
import os

# Define command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Cheminformatics Model Training and Evaluation")
    #parser.add_argument("--data_path", type=str, required=True, help="Path to the dataset directory")
    parser.add_argument("--dataset_name", type=str, required=True, help="Name of the dataset file")
    parser.add_argument("--model", type=str, required=True, choices=['SVM', 'XGB', 'RF'], help="Model to use")
    #parser.add_argument("--n_jobs", type=int, default=32, help="Number of jobs to run in parallel")
    return parser.parse_args()


    
class MMPmodel:
    def __init__(self, args):
        self.args = args
        self.task_binary = True
        self.data = self.get_dataset(args.dataset_name)
        self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test = self.process_dataset(self.data)
        self.model = self.get_model(args.model)
        self.results = self.train_evaluate_model(self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test, self.model)

    def get_dataset(self, dataset_name):
        # Retrieve label names based on the dataset

        
        # Determine the dataset type based on its name and initialize accordingly
        if dataset_name in ['ToxCast', 'Tox21']:
            # Toxicity prediction datasets
            label_lists = retrieve_label_name_list(dataset_name)
            data = Tox(name = dataset_name, label_name = label_lists[0])
        elif dataset_name in ['ClinTox']:
            data = Tox(name = 'ClinTox')
        elif dataset_name in ['SARSCoV2_Vitro_Touret', 'HIV']:
            # HTS datasets
            data = HTS(name = dataset_name)
            #ADME datasets
        elif dataset_name in ['PAMPA_NCATS', 'HIA_Hou','Pgp_Broccatelli','BBB_Martins']:
            data = ADME(name = dataset_name)
        elif dataset_name in ['HydrationFreeEnergy_FreeSolv', 'Lipo', 'ESOL']:
            data = ADME(name = dataset_name)
        
        else:
            raise ValueError(f"Dataset {dataset_name} is not recognized.")

        return data

    def get_task_binary(self, Y):
        # Determine if the task is binary or regression
        self.task_binary = True if len(np.unique(Y)) == 2 else False

    def process_dataset(self, data):
        # Convert dataset_name to the expected case, e.g., uppercase
        #dataset_name = dataset_name.upper()  # Ensure dataset_name is uppercase
        
        # Get the split for the dataset
        split = data.get_split()

        # Get the SMILES and labels for the dataset
        X_train, y_train = split['train']['Drug'], split['train']['Y']
        X_val, y_val = split['valid']['Drug'], split['valid']['Y']
        X_test, y_test = split['test']['Drug'], split['test']['Y']

        #get pubchem fingerprints
        fp = FingerprintProcessor()
        X_train = fp.pubchem_fp(X_train)
        X_val = fp.pubchem_fp(X_val)
        X_test = fp.pubchem_fp(X_test)

        return X_train, X_val, X_test, y_train, y_val, y_test
        # print(X_train)
        # print(X_val)
        # print(type(X_train))
        # print("X_train shape is:", X_train.shape)
        # print("X_val shape is:", X_val.shape)
    '''
        #Create dataset with RDKit2D descriptors
        rdkit2d = RDKit2DNormalized()
        x_test_mol, X_train = rdkit2d.processSmiles(X_train)
        x_val_mol, X_val = rdkit2d.processSmiles(X_val)
        x_test_mol, X_test = rdkit2d.processSmiles(X_test)
        
        #Treat Nan values
        imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
        X_train = imputer.fit_transform(X_train)
        X_val = imputer.transform(X_val)
        X_test = imputer.transform(X_test)
        # try:
        #     print("X_train shape after is:", len(X_train))
        #     print("X_val shape after is:", len(X_val))
        # except TypeError:
        #     print("Error in shape")

        # sequences = X_train
        # Calculate the lengths of all sequences
        # lengths = [len(seq) for seq in sequences]
        # print("the length for X_train is:",lengths)
        # print("the length for X_val is:",len(X_val))
    '''   
        

    # # Main function
    # if __name__ == "__main__":
    #     args = parse_args()
    #     X_train, X_val, y_train, y_val = process_dataset(args.dataset_name)
    #     # model = get_model(args.model)
    #     # train_evaluate_model(X_train, X_test, y_train, y_test, model)
    #     print(X_val)

    # Define model based on user input
    def get_model(self, model_name):
        # Define the Classification model based on the model_name
        if self.task_binary:
            if model_name == 'XGB':
                return XGBClassifier(
                    n_estimators=100,
                    learning_rate=0.05,
                    max_depth=4,
                    colsample_bytree=0.7,
                    subsample=0.8
                )
            elif model_name == 'RF':
                return RandomForestClassifier(
                    bootstrap=True,
                    max_depth=10,
                    min_samples_leaf=1,
                    min_samples_split=2,
                    n_estimators=300
                )
            elif model_name == 'SVM':
                return SVC(
                    C=10,
                    coef0=0.0,
                    degree=2,
                    gamma='scale',
                    kernel='rbf'    
                )
            else:
                raise ValueError("Model not supported")
        # Define the Regression model based on the model_name
        else:
            if model_name == 'XGB':
                return XGBRegressor(
                    objective='reg:squarederror'
                )
            elif model_name == 'RF':
                return RandomForestRegressor(
                )
            elif model_name == 'SVM':
                return SVR(    
                )
            else:
                raise ValueError("Model not supported")
    # Train and evaluate model
    def train_evaluate_model(self, X_train, X_val, X_test, y_train, y_val, y_test, model):
        model.fit(X_train, y_train)
        # Evaluate on validation set
        if self.task_binary:
            predictions_val = model.predict(X_val)
            accuracy_val = accuracy_score(y_val, predictions_val)
            auc_val = roc_auc_score(y_val, predictions_val)

            # Evaluate on test set
            predictions_test = model.predict(X_test)
            accuracy_test = accuracy_score(y_test, predictions_test)
            auc_test = roc_auc_score(y_test, predictions_test)

            # Collect results
            results = {
                'Accuracy': {'Validation': accuracy_val, 'Test': accuracy_test},
                'AUC': {'Validation': auc_val, 'Test': auc_test}
            }    
            # Add more metrics as needed
            print(f"Validation Accuracy: {accuracy_val}, Validation AUC: {auc_val}")
            print(f"Test Accuracy: {accuracy_test}, Test AUC: {auc_test}")
            # print("Test Classification Report:")
            # print(classification_report(y_test, predictions_test))
        # Regression task
        else:
            predictions_val = model.predict(X_val)
            mse_val = mean_squared_error(y_val, predictions_val)
            mae_val = mean_absolute_error(y_val, predictions_val)
            r2_val = r2_score(y_val, predictions_val)

            # Evaluate on test set
            predictions_test = model.predict(X_test)
            mse_test = mean_squared_error(y_test, predictions_test)
            mae_test = mean_absolute_error(y_test, predictions_test)
            r2_test = r2_score(y_test, predictions_test)

            # Collect results
            results = {
                'MSE': {'Validation': mse_val, 'Test': mse_test},
                'MAE': {'Validation': mae_val, 'Test': mae_test},
                'R2': {'Validation': r2_val, 'Test': r2_test}
            }
            # Add more metrics as needed
            print(f"Validation MSE: {mse_val}, Validation MAE: {mae_val}, Validation R2: {r2_val}")
            print(f"Test MSE: {mse_test}, Test MAE: {mae_test}, Test R2: {r2_test}")
        
        return results
        #save_results(results, args.dataset_name, args.model)

    def save_results(self, results, dataset_name, model, results_dir='pubchem_results'):
        """
        Saves the evaluation results to a CSV file.

        Parameters:
        - results: A dictionary containing the evaluation metrics.
        - filename: The name of the file where results will be saved.
        """
        # Ensure the directory exists
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        
        # Construct the filename
        filename = f"{dataset_name}_{model}.csv"
        file_path = os.path.join(results_dir, filename)

        # Write results to CSV
        with open(file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            # Write the headers
            writer.writerow(['Metric', 'Validation', 'Test'])
            # Write the data
            for key, value in results.items():
                writer.writerow([key, value['Validation'], value['Test']])
        print(f"Results saved to {file_path}")


    
    # Save results to CSV


# Note: Ensure X_test and y_test are passed to the function along with other parameters

# Main function
if __name__ == "__main__":
    args = parse_args()
    mmp_model = MMPmodel(args)
    dataset = mmp_model.get_dataset(args.dataset_name)
    X_train, X_val, X_test, y_train, y_val, y_test = mmp_model.process_dataset(dataset)
    mmp_model.get_task_binary(y_train)
    model = mmp_model.get_model(args.model)
    results = mmp_model.train_evaluate_model(X_train, X_val, X_test, y_train, y_val, y_test, model)
    mmp_model.save_results(results, args.dataset_name, args.model)

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