import random

import numpy as np
<<<<<<< HEAD
from utils import DatasetLoader, ModelTuner, save_results
import argparse
import os
from sklearn.metrics import root_mean_squared_error, roc_auc_score
=======
from utils import DatasetLoader, ModelTuner, MMPmodel
import argparse
import os
>>>>>>> origin/master

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

    seeds = [42, 1234, 7]
    results = []
    for seed in seeds:
        # tune hyperparameters for the model and dataset
        X_train, X_val, X_test, y_train, y_val, y_test = dataloader.split(X, Y, seed)
        tuner = ModelTuner(args.model, X_train, y_train, task_binary, seed, args.n_jobs)
        best_model, best_param = tuner.tune_para()
        # save parameters for the best model
        predictions_val = best_model.predict(X_test)
        if task_binary:
            auc = roc_auc_score(y_test, predictions_val)
            results.append(auc)
        else:
            rmse = root_mean_squared_error(y_test, predictions_val)
            results.append(rmse)
    if task_binary:
        auc_mean = np.mean(results)
        auc_std = np.std(results)
        rsl_save = {'AUC': {'mean': auc_mean, 'std': auc_std}}
    else:
        rmse_mean = np.mean(results)
        rmse_std = np.std(results)
        rsl_save = {'RMSE': {'mean': rmse_mean, 'std': rmse_std}}
    save_results(rsl_save, args.dataset_name, args.model, args.output_dir)



