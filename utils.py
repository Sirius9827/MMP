from math import sqrt
import numpy as np  
from sklearn.model_selection import KFold, ParameterGrid, StratifiedKFold, train_test_split
#from sklearn.svm import SVC
from sklearn.svm import SVC, SVR
from xgboost import XGBClassifier, XGBRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
# evaluation metrics for classification and regression
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error, mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

import pandas as pd  
#from descriptors.rdNormalizedDescriptors import RDKit2DNormalized
from allfingerprints import FingerprintProcessor

import csv
import os
from rdkit import RDLogger

# Suppress RDKit warnings
RDLogger.DisableLog('rdApp.*')


class DatasetLoader:
    def __init__(self, root_path, dataset):
        self.dataset_path = os.path.join(root_path, f"{dataset}/{dataset}.csv")
        # Load Data
        self.df = pd.read_csv(self.dataset_path)
        self.smiles = self.df['smiles']
        self.labels = np.ravel(self.df.drop(columns=['smiles']))
        self.task_names = self.df.columns.drop(['smiles']).tolist()
        self.n_tasks = len(self.task_names)

    def featurize(self):
        fp = FingerprintProcessor()
        X_features = fp.concat_fp(self.smiles)
        X_features = np.array(X_features)
        return X_features
    
    def orthogonal(self):
        fp = FingerprintProcessor()
        X_features = fp.orth_fp(self.smiles)
        X_features = np.array(X_features)
        return X_features

    def split(self, X, Y, seed):
        
        X_train, X_temp, y_train, Y_temp = train_test_split(X, Y, test_size=0.2, random_state=seed)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, Y_temp, test_size=0.5, random_state=seed)
        
        return X_train, X_val, X_test, y_train, y_val, y_test

    def task_binary(self, Y):
        task_binary = len(np.unique(Y)) == 2
        return task_binary
    
class MMPmodel:
    def __init__(self, model_name, X, Y, task_binary, seed):
        self.X = X
        self.Y = Y
        # self.X_val = X_val
        # self.y_val = y_val
        self.model_name = model_name
        self.task_binary = task_binary
        self.seed = seed

    # Define model based on user input
    def get_model(self, model_name, seed):
        # Define the Classification model based on the model_name
        if self.task_binary:
            if model_name == 'XGB':
                return XGBClassifier(
                    tree_method='hist',
                    device = 'cuda',
                    random_state=seed,
                    n_jobs=-1
                )
            elif model_name == 'RF':
                return RandomForestClassifier(
                    random_state=seed,
                    n_jobs=-1
                )
            elif model_name == 'SVM':
                return SVC(   
                    # probability=True,
                    # random_state=seed,
                )
            else:
                raise ValueError("Model not supported")
        # Define the Regression model based on the model_name
        else:
            if model_name == 'XGB':
                return XGBRegressor(
                    objective='reg:squarederror',
                    random_state=seed,
                    n_jobs=-1
                )
            elif model_name == 'RF':
                return RandomForestRegressor(
                    random_state=seed,
                    n_jobs=-1
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

    def tune_para(self):
        mmp = MMPmodel(self.model, self.X, self.Y, self.X_val, self.y_val, self.task_binary, self.seed)
        get_mdl = mmp.get_model(self.model, self.seed)
        if self.model == 'SVM':
             param_grid = {
                'C': [0.1, 1, 10, 100],
                'gamma': [0.2, 0.1, 0.01, 0.001],
                'kernel': ['rbf'],
            }
        elif self.model == 'XGB':
            # Tune the parameters for XGB
            param_grid = {
                'n_estimators': [50, 100, 200, 300, 400, 500, 1000],
                'max_depth': [3, 5, 7, 9, 10],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.7, 0.8, 0.9, 1.0],
                'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
                'gamma': [0, 0.1, 0.2],
                'min_child_weight': [1, 3, 6],
                }
            
        elif self.model == 'RF':
            # Tune the parameters for RF
            param_grid = {
                'n_estimators': [50, 100, 200, 300, 400, 500, 1000],
                'max_features': ['sqrt', 'log2', 0.7, 0.8, 0.9],
                'max_depth': [3, 6, 8, 10, 12],
                'min_samples_leaf': [1, 3, 5, 10, 20, 50],
                'min_impurity_decrease': [0.0, 0.01],
            }
            if self.task_binary:
                param_grid['criterion'] = ['gini', 'entropy']
            else:
                param_grid['criterion'] = ['squared_error']

        param_combinations = list(ParameterGrid(param_grid))
        best_score = -np.inf if self.task_binary else np.inf
        score = []
        for params in param_combinations:
            get_mdl.set_params(**params)
            get_mdl.fit(self.X, self.Y)
            predictions_val = get_mdl.predict(self.X_val)
            if self.task_binary:
                auc = roc_auc_score(self.y_val, predictions_val)
                if auc > best_score:
                    best_score = auc
                    best_params = params
                    y_pred = get_mdl.predict(self.X_test)

            else:
                rmse = root_mean_squared_error(self.y_val, predictions_val)
                if rmse < best_score:
                    best_score = rmse
                    best_params = params
                    y_pred = get_mdl.predict(self.X_test)
            

        # cv_results = []
        auc = make_scorer(roc_auc_score, needs_proba=True, greater_is_better=True)
        rmse = make_scorer(mean_squared_error, greater_is_better=False)




        # best_score = -np.inf if self.task_binary else np.inf

        # if self.task_binary:
        #     # Initialize GridSearchCV with AUC as the scoring metric
        #     skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.seed)
        #     grid_search = GridSearchCV(estimator=get_mdl, param_grid=param_grid, scoring=auc, cv=skf, verbose=1, n_jobs=self.n_jobs)
        # else:
        #     # Initialize GridSearchCV with RMSE as the scoring metric
        #     kf = KFold(n_splits=5, shuffle=True, random_state=self.seed)
        #     grid_search = GridSearchCV(estimator=get_mdl, param_grid=param_grid, scoring=rmse, cv=kf, verbose=1, n_jobs=self.n_jobs)
        # print(f"the model is {get_mdl}")
        # grid_search.fit(self.X, self.Y)
        # cv_results.append(grid_search.cv_results_['mean_test_score'])

        # # Calculate the average scores across different seeds
        # avg_scores = np.mean(cv_results, axis=0)
        # best_index = np.argmax(avg_scores)
        # best_params = grid_search.cv_results_['params'][best_index]
        # # best_model = get_mdl(param_grid=best_params)

        # print(f"Best parameters: {best_params}")
        # print(f"Best average score: {avg_scores[best_index]}")
        
        # Return the best estimator
        # return grid_search.best_estimator_, grid_search.best_params_
        return auc, rmse

    def tune_randomized_search(self, X, Y, get_mdl, param_grid, task_binary, seed, n_iter=1000, n_jobs=-1):
        best_score = -np.inf if task_binary else np.inf

        if task_binary:
            # Initialize RandomizedSearchCV with AUC as the scoring metric
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
            random_search = RandomizedSearchCV(estimator=get_mdl, param_distributions=param_grid, scoring='roc_auc', cv=skf, verbose=1, n_jobs=n_jobs, n_iter=n_iter, random_state=seed)
        else:
            # Initialize RandomizedSearchCV with RMSE as the scoring metric
            kf = KFold(n_splits=5, shuffle=True, random_state=seed)
            random_search = RandomizedSearchCV(estimator=get_mdl, param_distributions=param_grid, scoring='neg_root_mean_squared_error', cv=kf, verbose=1, n_jobs=n_jobs, n_iter=n_iter, random_state=seed)

        random_search.fit(X, Y)
        best_params = random_search.best_params_
        best_model = random_search.best_estimator_

        print(f"Best parameters: {best_params}")
        print(f"Best score: {random_search.best_score_}")

        return best_model, best_params

    # Example usage
    # best_model, best_params = tune_randomized_search(X, Y, get_mdl, param_grid, task_binary, seed)


def save_results(results, dataset_name, model, results_dir):
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
        writer.writerow(['Metric', 'Mean', 'Std'])
        # Write the data
        for key, value in results.items():
            writer.writerow([key, value['mean'], value['std']])
    print(f"Results saved to {file_path}")
