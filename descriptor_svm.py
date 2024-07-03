import numpy as np  
from rdkit import Chem  
from rdkit.Chem import AllChem 
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import mean_squared_error
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

for seed in [1, 2, 3, 4, 5]:
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=seed)
    predictions = {}

    #Train SVM Model
    svm_model = SVC(kernel='linear')  # You can adjust the kernel and other parameters
    svm_model.fit(X_train, y_train)

    y_pred_test = svm_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred_test)

    print("Accuracy:", accuracy)  
    print("Classification Report:")  
     
    predictions[name] = y_pred_test
    predictions_list.append(predictions)

#results = group.evaluate_many(predictions_list)