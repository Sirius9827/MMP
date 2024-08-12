import argparse
import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier, MLPRegressor
import torch
from pretrain_trfm import TrfmSeq2seq
from pretrain_rnn import RNNSeq2Seq
# from bert import BERT
from build_vocab import WordVocab
from utils import split

def args():
    parser = argparse.ArgumentParser(description='Build a corpus file')
    parser.add_argument('--in_path', '-i', type=str, default='data/chembl_24.csv', help='input file')
    parser.add_argument('--out_path', '-o', type=str, default='data/chembl24_corpus.txt', help='output file')
    args = parser.parse_args()
    return args

def get_inputs(sm):
    seq_len = 220
    sm = sm.split()
    if len(sm)>218:
        print('SMILES is too long ({:d})'.format(len(sm)))
        sm = sm[:109]+sm[-109:]
    ids = [vocab.stoi.get(token, unk_index) for token in sm]
    ids = [sos_index] + ids + [eos_index]
    seg = [1]*len(ids)
    padding = [pad_index]*(seq_len - len(ids))
    ids.extend(padding), seg.extend(padding)
    return ids, seg

def get_array(smiles):
    x_id, x_seg = [], []
    for sm in smiles:
        a,b = get_inputs(sm)
        x_id.append(a)
        x_seg.append(b)
    return torch.tensor(x_id), torch.tensor(x_seg)

def evaluate_mlp_regression(X, y, rate, n_repeats):
    r2, rmse = np.empty(n_repeats), np.empty(n_repeats)
    for i in range(n_repeats):
        reg = MLPRegressor(max_iter=1000)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1-rate)
        reg.fit(X_train, y_train)
        y_pred = reg.predict(X_test)
        r2[i] = r2_score(y_pred, y_test)
        rmse[i] = mean_squared_error(y_pred, y_test)**0.5
    ret = {}
    ret['r2 mean'] = np.mean(r2)
    ret['r2 std'] = np.std(r2)
    ret['rmse mean'] = np.mean(rmse)
    ret['rmse std'] = np.std(rmse)
    return ret

def evaluate_mlp_classification(X, y, rate, n_repeats):
    auc = np.empty(n_repeats)
    for i in range(n_repeats):
        clf = MLPClassifier(max_iter=1000)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1-rate, stratify=y)
        clf.fit(X_train, y_train)
        y_score = clf.predict_proba(X_test)
        auc[i] = roc_auc_score(y_test, y_score[:,1])
    ret = {}
    ret['auc mean'] = np.mean(auc)
    ret['auc std'] = np.mean(np.std(auc, axis=0))
    return ret

rates = 2**np.arange(7)/80

pad_index = 0
unk_index = 1
eos_index = 2
sos_index = 3
mask_index = 4

vocab = WordVocab.load_vocab('data/vocab.pkl')
trfm = TrfmSeq2seq(len(vocab), 256, len(vocab), 4)
trfm.load_state_dict(torch.load('data/trfm_12_23000.pkl'))

trfm.eval()
# print('total parameters:', sum(p.numel() for p in trfm.parameters()))
args = args()
input_file = args.in_path
output_file = args.out_path
data_file = os.path.join('data/datasets', input_file, f"{input_file}.csv")

df = pd.read_csv(data_file)
x_split = [split(sm) for sm in df['smiles'].values]
xid, xseg = get_array(x_split)
X = trfm.encode(torch.t(xid))
print("The shape of X is",X.shape)

for rate in rates:
    score_dic = evaluate_mlp_classification(X, df.iloc[:, 1].values, rate, 20)
    print(rate, score_dic)