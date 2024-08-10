### allfingerprint.py 
generates fingerprints for corresponding smiles
### prediction_ML.py 
includes module for dataloading, precessing, finetuning and test&evaluating
usage
```python
python prediction_ML.py -d dataset -m method -o outputdir
```
similar usage include predict_orthogonal.py, which is used for ML prediction for orthogonal transformations on datasets features
predict_randomcv.py, used for using random search for optimizing parameter
### run_MLP
```python
python run_MLP.py -d dataset -o outputdir
```
### multitask.py 
is for spliting multi-task prediction to single-task prediction and get average metric score

