import numpy as np  
from rdkit import Chem  
from rdkit.Chem import AllChem 
from sklearn.model_selection import train_test_split
#from sklearn.svm import SVC
from xgboost import XGBClassifier 
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import mean_squared_error #for regression task
from sklearn.metrics import mean_absolute_error
import pandas as pd  
import pandas_flavor as pf  
from descriptors import *
from descriptors.rdDescriptors import RDKit2D
from sklearn.impute import SimpleImputer 

train_val = "data/admet_group/ames/train_val.csv"
test = "data/admet_group/ames/test.csv"
train_val = pd.read_csv(train_val)
test = pd.read_csv(test)

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