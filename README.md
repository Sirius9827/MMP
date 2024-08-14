## This script is used for molecule property prediction on comprehensive benchmarks
### prediction_ML.py 
includes module for dataloading, precessing, model finetuning and test&evaluating. Supported method includes SVM, RF, XGB
usage
```python
python prediction_ML.py -d dataset -m method -o outputdir
```
similar usage include predict_orthogonal.py, which is used for ML prediction for orthogonal transformations on datasets features
usage
```python
python predict_orthogonal.py -d dataset -m method -o outputdir
```
predict_randomcv.py, is explicitly used for XGB optimizing parameter using random search and cross validation in the paper, as the search space is huge.
### run_MLP
using MLP to predict molecule property, the input features are the same as mentioned above
```python
python run_MLP.py -d dataset -o outputdir
```
### allfingerprint.py 
generates fingerprints for corresponding smiles

### multitask.py 
is for spliting multi-task prediction to single-task prediction and get average metric score

### trsf_mmp.py
Please copy vocab.pkl and trfm_12_23000.pkl to smiles_transformer/data file folder
```python
python trsf_mmp.py -i dataset.csv
```

