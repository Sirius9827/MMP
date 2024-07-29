import random

import numpy as np
from utils import DatasetLoader, ModelTuner, MMPmodel
import argparse
import os

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

    # tune hyperparameters for the model and dataset
    tuner = ModelTuner(args.model, X, Y, task_binary, args.n_jobs)
    best_model, best_param = tuner.tune_para()
    # save parameters for the best model
    
    
    # Generate 20 different random seeds
    seeds = [random.randint(0, 10000) for _ in range(20)]
    premodel = MMPmodel(args.model, X, Y, task_binary)
    all_results = []
    results_list = []
    # evaluate the model with 20 different random seeds
    for seed in seeds:
        X_train, X_val, X_test, y_train, y_val, y_test = dataloader.split(X, Y, seed)
        results = premodel.train_evaluate_model(X_train, X_val, X_test, y_train, y_val, y_test, best_model)
        results_list.append(results)
    # Collect AUC and RMSE test scores
    if task_binary:
        auc_test_scores = [result['AUC']['Test'] for result in results_list]
        auc_mean = np.mean(auc_test_scores)
        auc_std = np.std(auc_test_scores)
        rsl_save = {'AUC': {'mean': auc_mean, 'std': auc_std}}
    else:
        rmse_test_scores = [result['RMSE']['Test'] for result in results_list]
        rmse_mean = np.mean(rmse_test_scores)
        rmse_std = np.std(rmse_test_scores)
        rsl_save = {'RMSE': {'mean': rmse_mean, 'std': rmse_std}}
    premodel.save_results(rsl_save, args.dataset_name, args.model, args.output_dir)


