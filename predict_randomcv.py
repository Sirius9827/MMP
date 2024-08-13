import random

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from utils import DatasetLoader, MMPmodel, save_results
import argparse
import os
from sklearn.metrics import root_mean_squared_error, roc_auc_score
from sklearn.model_selection import ParameterGrid, train_test_split
from sklearn.metrics import roc_auc_score, mean_squared_error
import numpy as np

# Define command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Cheminformatics Model Training and Evaluation")
    #parser.add_argument("--data_path", type=str, required=True, help="Path to the dataset directory")
    parser.add_argument("-d", "--dataset_name", type=str, required=True, help="Name of the dataset file")
    parser.add_argument("-m","--model", type=str, required=True, choices=['SVM', 'XGB', 'RF', 'all'], help="Model to use")
    parser.add_argument("-j","--n_jobs", type=int, default=-1, help="Number of jobs to run in parallel.")
    parser.add_argument("-o","--output_dir", type=str, default="results", help="Directory to save results")
    #parser.add_argument("--n_jobs", type=int, default=32, help="Number of jobs to run in parallel")
    return parser.parse_args()

if __name__ == '__main__':

    args = parse_args()
    dataloader = DatasetLoader("datasets", args.dataset_name)
    X = dataloader.featurize()
    Y = dataloader.labels
    task_binary = dataloader.task_binary(Y)
    print("Task is binary: ", task_binary)

    # Assuming self.model is 'SVM', 'XGB', or 'RF'
    if args.model == 'SVM':
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'gamma': [0.2, 0.1, 0.01, 0.001],
            'kernel': ['rbf'],
        }
    elif args.model == 'XGB':
        param_grid = {
            'n_estimators': [50, 100, 200, 300, 400, 500, 1000],
            'max_depth': [3, 5, 7, 9, 10],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.7, 0.8, 0.9, 1.0],
            'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
            'gamma': [0, 0.1, 0.2],
            'min_child_weight': [1, 3, 6],
        }
    elif args.model == 'RF':
        param_grid = {
            'n_estimators': [50, 100, 200, 300, 400, 500, 1000],
            'max_features': ['sqrt', 'log2', 0.7, 0.8, 0.9],
            'max_depth': [3, 6, 8, 10, 12],
            'min_samples_leaf': [1, 3, 5, 10, 20, 50],
            'min_impurity_decrease': [0.0, 0.01],
        }
        if task_binary:
            param_grid['criterion'] = ['gini', 'entropy']
        else:
            param_grid['criterion'] = ['squared_error']


    best_params = None
    best_std = None

    # Split the data into 3 random splits
    if task_binary:
        seeds = [42, 1234, 7]
    # for regression tasks we use 10 random splits to get reasonable estimates of standard deviation
    else:
        seeds = [42, 1234, 7, 100, 200, 300, 400, 500, 600, 700]
        
    results = []
    for seed in seeds:
        # Initialize the model 
        mmp = MMPmodel(args.model,X, Y, task_binary, seed)
        model = mmp.get_model(args.model, seed)
        # split the dataset
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=seed)
        # tune the model using randomized search, set iterations
        tuner = MMPmodel(args.model, X_train, y_train, task_binary, seed)
        # save parameters for the best model
        best_model, best_param = tuner.tune_randomized_search(X_train, y_train, model, param_grid, task_binary, seed, n_iter=200)
        # get the predictions on test set
        predictions_val = best_model.predict(X_test)
        if task_binary:
            auc = roc_auc_score(y_test, predictions_val)
            results.append(auc)
        else:
            rmse = root_mean_squared_error(y_test, predictions_val)
            results.append(rmse)
    # get the mean and std of the results
    if task_binary:
        auc_mean = np.mean(results)
        auc_std = np.std(results)
        rsl_save = {'AUC': {'mean': auc_mean, 'std': auc_std}}
    else:
        rmse_mean = -np.mean(results)
        rmse_std = np.std(results)
        rsl_save = {'RMSE': {'mean': rmse_mean, 'std': rmse_std}}
    save_results(rsl_save, args.dataset_name, args.model, args.output_dir)
        





