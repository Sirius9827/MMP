import argparse
from collections import defaultdict
from math import sqrt
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
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, roc_auc_score

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
    parser.add_argument("--model", type=str, required=True, choices=['SVM', 'XGB', 'RF', 'all'], help="Model to use")
    parser.add_argument("--n_jobs", type=int, default=1, help="Number of jobs to run in parallel.")
    #parser.add_argument("--n_jobs", type=int, default=32, help="Number of jobs to run in parallel")
    return parser.parse_args()

class DatasetLoader:
    def __init__(self):
        self.dataset_paths = {
            'lipop': 'datasets/lipo/lipo.csv',
            'bace': 'datasets/bace/bace.csv',
            'bbbp': 'datasets/bbbp/bbbp.csv',
            'esol': 'datasets/esol/esol.csv',
            'sider': 'datasets/sider/sider.csv'
        }
        self.y_labels = {
            'lipop': 'lipo',
            'bace': 'Class',
            'bbbp': 'p_np',
            'esol': 'logSolubility',
            'sider': 'multiple label'  # Placeholder, replace with actual Y label for 'sider'
        }

    def get_dataset(self, dataset_name):
        if dataset_name in self.dataset_paths:
            data = pd.read_csv(self.dataset_paths[dataset_name])
        else:
            data = self.initialize_custom_dataset(dataset_name)
        return data

    def initialize_custom_dataset(self, dataset_name):
        # Initialize custom datasets based on dataset_name
        if dataset_name in ['ToxCast', 'Tox21']:
            # Toxicity prediction datasets
            label_lists = retrieve_label_name_list(dataset_name)
            num_labels = len(label_lists)
            print(f"Number of labels in {dataset_name}: {num_labels}")

            return Tox(name=dataset_name, label_name=label_lists[0])
        elif dataset_name in ['ClinTox']:
            return Tox(name='ClinTox')
        elif dataset_name in ['SARSCoV2_Vitro_Touret', 'HIV']:
            # HTS datasets
            return HTS(name=dataset_name)
        elif dataset_name in ['PAMPA_NCATS', 'HIA_Hou', 'Pgp_Broccatelli', 'BBB_Martins', 'HydrationFreeEnergy_FreeSolv', 'ESOL']:
            # ADME datasets
            return ADME(name=dataset_name)
        else:
            # If dataset_name does not match any known datasets, return None or handle as appropriate
            return None

    def process_dataset(self, dataset_name, data):
        if dataset_name in ['lipop', 'bace', 'bbbp', 'esol', 'sider']:
            X = data['smiles']
            # Use the dataset_name to get the correct Y label from the self.y_labels dictionary
            Y_label = self.y_labels[dataset_name]
            # Handle the case where there are multiple labels for a dataset
            if isinstance(Y_label, list):
                Y = data[Y_label]
            else:
                Y = data[Y_label]
            X_train, X_temp, y_train, Y_temp = train_test_split(X, Y, test_size=0.2, random_state=42)
            X_val, X_test, y_val, y_test = train_test_split(X_temp, Y_temp, test_size=0.5, random_state=1234)
        else:
            # Assuming custom datasets have a method get_split()
            split = data.get_split(seed=42, frac = [0.8, 0.1, 0.1])
            X_train, y_train = split['train']['Drug'], split['train']['Y']
            X_val, y_val = split['valid']['Drug'], split['valid']['Y']
            X_test, y_test = split['test']['Drug'], split['test']['Y']

        # Process fingerprints for all datasets uniformly
        fp = FingerprintProcessor()
        X_train = fp.concat_fp(X_train)
        X_val = fp.concat_fp(X_val)
        X_test = fp.concat_fp(X_test)

        return X_train, X_val, X_test, y_train, y_val, y_test

    def get_task_binary(self, Y):
        self.task_binary = len(np.unique(Y)) == 2
    
class MMPmodel:
    def __init__(self, args):
        self.args = args
        self.task_binary = True
        # self.data = self.get_dataset(args.dataset_name)
        self.X_train = X_train
        self.X_val = X_val
        self.X_test = X_test
        self.y_train = y_train
        self.y_val = y_val
        self.y_test = y_test
        # self.model = self.get_model(model)
        # self.results = self.train_evaluate_model(self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test, self.model)

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
            rmse_val = sqrt(mse_val)  # Calculate RMSE for validation set
            mae_val = mean_absolute_error(y_val, predictions_val)
            r2_val = r2_score(y_val, predictions_val)

            # Evaluate on test set
            predictions_test = model.predict(X_test)
            mse_test = mean_squared_error(y_test, predictions_test)
            rmse_test = sqrt(mse_test)
            mae_test = mean_absolute_error(y_test, predictions_test)
            r2_test = r2_score(y_test, predictions_test)

            # Collect results
            results = {
                'MSE': {'Validation': mse_val, 'Test': mse_test},
                'RMSE': {'Validation': rmse_val, 'Test': rmse_test}, 
                'MAE': {'Validation': mae_val, 'Test': mae_test},
                'R2': {'Validation': r2_val, 'Test': r2_test}
            }
            # Add more metrics as needed
            print(f"Validation MSE: {mse_val}, Validation RMSE: {rmse_val}, Validation MAE: {mae_val}, Validation R2: {r2_val}")
            print(f"Test MSE: {mse_test}, Test RMSE: {rmse_test}, Test MAE: {mae_test}, Test R2: {r2_test}")
        
        return results
        #save_results(results, args.dataset_name, args.model)

    def save_results(self, results, dataset_name, model, results_dir='results1'):
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
            writer.writerow(['Metric', 'Test'])
            # Write the data
            for key, value in results.items():
                writer.writerow([key, value['Test']])
        print(f"Results saved to {file_path}")


class ModelTuner:
    def __init__(self, model, param_grid, X_train, y_train, X_val, y_val, n_jobs=1):
        self.model = model
        self.param_grid = param_grid
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.n_jobs = n_jobs
        self.task_binary = True

    def tune_parameters(self):
        if self.task_binary:
            # Define the AUC scorer
            auc_scorer = make_scorer(roc_auc_score, needs_proba=True, greater_is_better=True)
            # Initialize GridSearchCV
            grid_search = GridSearchCV(estimator=self.model, param_grid=self.param_grid, scoring=auc_scorer, cv=5, verbose=1, n_jobs=self.n_jobs)
        # Initialize GridSearchCV with MSE scorer
        else:
            # Define the MSE scorer
            mse_scorer = make_scorer(mean_squared_error, greater_is_better=False)
            grid_search = GridSearchCV(estimator=self.model, param_grid=self.param_grid, scoring=mse_scorer, cv=5, verbose=1, n_jobs=self.n_jobs)
    
        # Fit GridSearchCV
        grid_search.fit(self.X_train, self.y_train)
        
        # Print the best parameters and the best score
        print("Best Parameters:", grid_search.best_params_)
        if self.task_binary:
            print("Best AUC Score:", grid_search.best_score_)
        else:
            print("Best MSE Score:", -grid_search.best_score_)  # Multiply by -1 to convert back to positive value
    
        
        # Return the best estimator
        return grid_search.best_estimator_


# Note: Ensure X_test and y_test are passed to the function along with other parameters

# Main function
if __name__ == "__main__":
    args = parse_args()
    # # for single task prediction datasets
    loader = DatasetLoader()
    dataset = loader.get_dataset(args.dataset_name)
    X_train, X_val, X_test, y_train, y_val, y_test = loader.process_dataset(args.dataset_name, dataset)
    mmp_model = MMPmodel(args)
    task_binary = True
    if args.model == 'all':
        for model in ['SVM', 'XGB', 'RF']:
            model_instance = mmp_model.get_model(model)  # Instantiate the model
            if model == 'SVM':
                # Tune the parameters for SVM
                param_grid = {
                    'C': [0.1, 1, 10, 100],
                    'gamma': [0.2, 0.1, 0.01, 0.001],
                    'kernel': ['rbf'],
                    'probability': [True]
                }
                
            elif model == 'XGB':
                # Tune the parameters for XGB
                param_grid = {
                    'n_estimators': [50, 100, 200, 300, 400, 500, 1000],
                    'max_depth': [3, 4, 5, 6, 7, 8, 9, 10],
                    'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
                    'subsample': [0.7, 0.8, 0.9, 1.0],
                    'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
                    'colsample_bylevel': [0.7, 0.8, 0.9],
                    #add more parameters as needed
                    'gamma': [0, 0.1, 0.2],
                    'min_child_weight': [1, 2, 3, 4, 5, 6],
                    }
                
            elif model == 'RF':
                # Tune the parameters for RF
                if task_binary:
                    param_grid = {
                        'n_estimators': [50, 100, 200, 300, 400, 500, 1000],
                        'max_features': ['sqrt', 'log2', 0.7, 0.8, 0.9],
                        'max_depth': [3, 4, 6, 8, 10, 12],
                        'min_samples_leaf': [1, 3, 5, 10, 20, 50],
                        'min_inpurity_decrease': [0.0, 0.01],
                        'criterion': ['gini', 'entropy' if task_binary else 'mse']
                    }

            tuner = ModelTuner(model_instance, param_grid, X_train, y_train, X_val, y_val, n_jobs=args.n_jobs)
            model_instance = tuner.tune_parameters()
            mmp_model = MMPmodel(args)
            results = mmp_model.train_evaluate_model(X_train, X_val, X_test, y_train, y_val, y_test, model_instance)
            mmp_model.save_results(results, args.dataset_name, model)

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