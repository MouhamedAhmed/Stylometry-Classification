
import pandas as pd
from sklearn import preprocessing, decomposition, model_selection, metrics, pipeline
from sklearn.model_selection import train_test_split
import torch
import numpy as np


def get_acc(y_true, y_hat):
    correct = 0
    for i in range(len(y_hat)):
        try:
            if(np.argmax(y_hat[i]) == y_true[i]):
                correct += 1
        except:
            if(torch.argmax(y_hat[i]) == y_true[i]):
                correct += 1
    return correct/len(y_hat)


def read_data():
    train = pd.read_csv('./data.csv')

    lbl_enc = preprocessing.LabelEncoder()
    y = lbl_enc.fit_transform(train.author.values)

    xtrain, xvalid, ytrain, yvalid = train_test_split(train.text.values, y, 
                                                    stratify=y, 
                                                    random_state=42, 
                                                    test_size=0.1, shuffle=True)
    return xtrain, ytrain, xvalid, yvalid 
    

def convert_sparse_to_torch(xtrain, device):
    # convert to coo matrix
    coo = xtrain.tocoo()
    values = coo.data
    indices = np.vstack((coo.row, coo.col))
    # convert to torch sparse tensor
    values = torch.FloatTensor(values)
    indices = torch.LongTensor(indices)
    Features = torch.sparse.FloatTensor(indices, values, torch.Size(coo.shape)).to_dense().to(device)
    return Features
