print(1)

import argparse
from collections import defaultdict
print(2)

import numpy as np
print(3)
from property_prediction.DatasetLoader import process_dataset

print(4)
from tdc.utils import retrieve_label_name_list
print(5)
from tdc.single_pred import Tox
print(6)
# Define command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Cheminformatics Model Training and Evaluation")
    #parser.add_argument("--data_path", type=str, required=True, help="Path to the dataset directory")
    parser.add_argument("--dataset_name", type=str, required=True, help="Name of the dataset file")
    parser.add_argument("--model", type=str, required=True, choices=['SVM', 'XGB', 'RF', 'all'], help="Model to use")
    parser.add_argument("--n_jobs", type=int, default=1, help="Number of jobs to run in parallel.")
    return parser.parse_args()

# def to_submission_format(results):
#     df = pd.DataFrame(results)
#     def get_metric(x):
#         metric = []
#         for i in x:
#             metric.append(list(i.values())[0])
#         return [round(np.mean(metric), 3), round(np.std(metric), 3)]
#     return dict(df.apply(get_metric, axis = 1))

# Main function
print(3)
args = parse_args()
# # for single task prediction datasets
loader = DatasetLoader()
# dataset = loader.get_dataset(args.dataset_name)
# X_train, X_val, X_test, y_train, y_val, y_test = loader.process_dataset(args.dataset_name, dataset)
# mmp_model = MMPmodel(args)
task_binary = True
# for multiple labels
label_lists = retrieve_label_name_list(args.dataset_name)
num_labels = len(label_lists)
print(f"Number of labels in {args.dataset_name}: {num_labels}")
result_dict = {}
for label in label_lists:
    print(f"Processing label: {label}")
    dataset = Tox(name=args.dataset_name, label_name=label)
    X_train, X_val, X_test, y_train, y_val, y_test = loader.process_dataset(args.dataset_name, dataset)
    # dataset = mmp_model.get_dataset(args.dataset_name)
    # X_train, X_val, X_test, y_train, y_val, y_test = mmp_model.process_dataset(dataset)
    # mmp_model.task_binary(y_train)
    if args.model == 'all':
        for model in ['SVM', 'XGB', 'RF']:
            model_instance = mmp_model.get_model(model)  # Instantiate the model
            if model == 'SVM':
                # Tune the parameters for SVM
                param_grid = {
                    'C': [0.1, 1, 10, 100],
                    'gamma': [0.2, 0.1, 0.01, 0.001],
                    'kernel': ['rbf']
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
            result_dict[label].append(results)

    # print(to_submission_format(result_dict))


    # model = mmp_model.get_model(args.model)
    # param_grid = {
    # 'n_estimators': [100, 200, 500],
    # 'max_features': ['auto', 'sqrt'],
    # 'max_depth': [4, 6, 8],
    # 'criterion': ['squared_error', 'absolute_error', 'poisson', 'friedman_mse']  # Corrected criteria
    # }
    # tuner = ModelTuner(model, param_grid, X_train, y_train, X_val, y_val)
    # # best_model = tuner.tune_parameters()
    # results = mmp_model.train_evaluate_model(X_train, X_val, X_test, y_train, y_val, y_test, model)
    # mmp_model.save_results(results, args.dataset_name, args.model)
