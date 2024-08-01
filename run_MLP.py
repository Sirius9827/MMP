import argparse
# import deepchem as dc
import numpy as np
from sklearn.metrics import mean_squared_error, roc_auc_score
from sklearn.model_selection import train_test_split
from utils import DatasetLoader, save_results

import torch
import torch.nn as nn
import torch.optim as optim
from hyperopt import hp, fmin, tpe, Trials

def parse_args():
    parser = argparse.ArgumentParser(description="Cheminformatics Model Training and Evaluation")
    #parser.add_argument("--data_path", type=str, required=True, help="Path to the dataset directory")
    parser.add_argument("-d", "--dataset_name", type=str, required=True, help="Name of the dataset file")
    parser.add_argument("-m","--model", type=str, default="MLP", help="Model to use")
    parser.add_argument("-j","--n_jobs", type=int, default=-1, help="Number of jobs to run in parallel.")
    parser.add_argument("-o","--output_dir", type=str, default="results", help="Directory to save results")
    #parser.add_argument("--n_jobs", type=int, default=32, help="Number of jobs to run in parallel")
    return parser.parse_args()

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

args = parse_args()
dataloader = DatasetLoader("datasets", args.dataset_name)
X = dataloader.orthogonal()
Y = dataloader.labels
task_binary = dataloader.task_binary(Y)
print("Task is binary: ", task_binary)

# Define the MLP model class
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_rate, l2_reg):
        super(MLP, self).__init__()
        self.hidden = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.output = nn.Linear(hidden_size, 1)
        self.l2_reg = l2_reg

    def forward(self, x):
        x = self.hidden(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.output(x)
        return x

# Define the objective function for hyperparameter optimization
def objective(params):
    input_size = X.shape[1]  
    hidden_size = params['hidden_size']
    dropout_rate = params['dropout_rate']
    l2_reg = params['l2_reg']

    model = MLP(input_size, hidden_size, dropout_rate, l2_reg).to(device)
    if task_binary:
        criterion = nn.BCEWithLogitsLoss()
    else:  # regression
        criterion = nn.MSELoss()
    optimizer = optim.Adadelta(model.parameters(), weight_decay=l2_reg)

    seeds = [42, 1234, 7]  
    all_test_losses = []
    for seed in seeds:

      # Split the data into training and validation sets
      X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.2, random_state=seed)
      X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size=0.5, random_state=seed)

      # Convert data to PyTorch tensors
      X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
      y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(device)
      X_val = torch.tensor(X_val, dtype=torch.float32).to(device)
      y_val = torch.tensor(y_val, dtype=torch.float32).view(-1, 1).to(device)
      X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
      y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1).to(device)

    
      model.train()
      for epoch in range(50):  
          optimizer.zero_grad()
          outputs = model(X_train)
          loss = criterion(outputs, y_train)
          loss.backward()
          optimizer.step()

      # Validation
      model.eval()
      with torch.no_grad():
          val_outputs = model(X_val)
          if task_binary:
              val_outputs = torch.sigmoid(val_outputs)
              val_loss = roc_auc_score(y_val.cpu().numpy(), val_outputs.cpu().numpy())
          else:  # regression
              val_loss = mean_squared_error(y_val.cpu().numpy(), val_outputs.cpu().numpy(), squared=False)  # RMSE

      # Test
      model.eval()
      with torch.no_grad():
          test_outputs = model(X_test)
          if task_binary:
              test_outputs = torch.sigmoid(test_outputs)
              test_loss = roc_auc_score(y_test.cpu().numpy(), test_outputs.cpu().numpy())
          else:  # regression
              test_loss = mean_squared_error(y_test.cpu().numpy(), test_outputs.cpu().numpy(), squared=False)  # RMSE
      all_test_losses.append(test_loss)
      print("Test set performance:", test_loss)

    if task_binary:
        auc_mean = np.mean(all_test_losses)
        auc_std = np.std(all_test_losses)
        rsl_save = {'AUC': {'mean': auc_mean, 'std': auc_std}}
    else:
        rmse_mean = np.mean(all_test_losses)
        rmse_std = np.std(all_test_losses)
        rsl_save = {'RMSE': {'mean': rmse_mean, 'std': rmse_std}}
    save_results(rsl_save, args.dataset_name, args.model, args.output_dir)

    print("Mean test set performance:", np.mean(all_test_losses))

    return val_loss if task_binary else -val_loss


    # Define the search space
search_space = {
    'hidden_size': hp.choice('hidden_size', [64, 128, 256, 512]),
    'dropout_rate': hp.uniform('dropout_rate', 0.0, 0.5),
    'l2_reg': hp.uniform('l2_reg', 0.0, 0.01)
}

# Run the hyperparameter optimization
trials = Trials()
best_params = fmin(fn=objective,
                   space=search_space,
                   algo=tpe.suggest,
                   max_evals=100,
                   trials=trials)

# Map the best hidden_size index to the actual value
hidden_sizes = [64, 128, 256, 512]
best_params['hidden_size'] = hidden_sizes[best_params['hidden_size']]

print("Best hyperparameters:", best_params)




# model = dc.models.TorchModel(pytorch_model, dc.models.losses.L2Loss())
# metric = dc.metrics.Metric(dc.metrics.rms_score)
# model.fit(train_dataset, nb_epoch=50)
# print('training set score:', model.evaluate(train_dataset, [metric]))
# print('test set score:', model.evaluate(test_dataset, [metric]))

# search_space = {
#     'layer_sizes': hp.choice('layer_sizes',[[500], [1000], [2000],[1000,1000]]),
#     'dropouts': hp.uniform('dropout',low=0.2, high=0.5),
#     'learning_rate': hp.uniform('learning_rate',high=0.001, low=0.0001)
# }
# import tempfile
# #tempfile is used to save the best checkpoint later in the program.

# metric = dc.metrics.Metric(dc.metrics.roc_auc_score)

# def fm(args):
#   save_dir = tempfile.mkdtemp()
#   model = dc.models.MultitaskClassifier(n_tasks=len(tasks),n_features=1024,layer_sizes=args['layer_sizes'],dropouts=args['dropouts'],learning_rate=args['learning_rate'])
#   #validation callback that saves the best checkpoint, i.e the one with the maximum score.
#   validation=dc.models.ValidationCallback(valid_dataset, 1000, [metric],save_dir=save_dir,transformers=transformers,save_on_minimum=False)
  
#   model.fit(train_dataset, nb_epoch=25,callbacks=validation)

#   #restoring the best checkpoint and passing the negative of its validation score to be minimized.
#   model.restore(model_dir=save_dir)
#   valid_score = model.evaluate(valid_dataset, [metric], transformers)

#   return -1*valid_score['roc_auc_score']

# tasks, datasets, transformers = dc.molnet.load_hiv(featurizer='ECFP', split='scaffold')
# train_dataset, valid_dataset, test_dataset = datasets
# trials=Trials()
# best = fmin(fm,
#     		space= search_space,
#     		algo=tpe.suggest,
#     		max_evals=15,
#     		trials = trials)