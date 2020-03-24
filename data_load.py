import numpy as np
import pandas as pd

import os
import yaml

import torch
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from imblearn.over_sampling import SMOTE, BorderlineSMOTE, SVMSMOTE
from imblearn.combine import *

def normalize(X, exist=None, mean=None, std=None):
    #rest_ind = [0, 4, 21, 22, 23, 27, 28, 29, 30, 32, 33, 34, 35, 36, 37, 38,
            #39, 40, 41, 42, 43, 44, 45, 46, 55, 59, 60] # exclude ASA, gender
    rest_ind = [4, 21, 22, 27, 28, 29, 30, 32, 33, 34, 35, 36, 37, 38,
            39, 40, 41, 43, 44, 45, 46, 55, 59, 60]
    X = np.copy(X)
    Y = X[:, rest_ind]
    if exist is not None:
        exist_Y = exist[:, rest_ind]
    else:
        exist_Y = np.ones_like(Y)

    if mean is None:
        mean = (Y * exist_Y).sum(axis=0, keepdims=True) / exist_Y.sum(axis=0, keepdims=True)
    if std is None:
        std = np.sqrt(np.square((Y - mean) * exist_Y).sum(axis=0, keepdims=True) / exist_Y.sum(axis=0, keepdims=True))
    Y = (Y - mean) / std * exist_Y
    X[:, rest_ind] = Y
    return X, mean, std

def from_numpy(*args):
    return (torch.from_numpy(arg) for arg in args)

def data_load_mysplit(n_splits=3, all_set=None,
        random_state=123, ensemble=False, oversampling='none'):
    '''
        oversampling : 'none', 'smote'
    '''
    if all_set is None:
        all_set = pd.read_excel("./ML용_codingbook_new강화평180709.xls", "All set")

    all_label = pd.get_dummies(all_set["DESAT"]).values
    all_data = all_set.iloc[:, 3:].values.astype(float) # SN, UN, DESAT Exclusion

    # TODO: to handle missing features appropriately
    all_exist = (all_data != 99899).astype(float) # For normalize

    all_data[all_data == 99899] = 0

    split_criteria = np.where(all_data[:, 48] > 2.5, 1, 0)
    # Standard normalization
    all_data, mean, std = normalize(all_data, all_exist)

    # Split data into train : val : test = 5 : 2 : 3
    train_data, test_data, train_exist, test_exist, train_label, test_label, train_split, _ = \
        train_test_split(all_data, all_exist, all_label, split_criteria, test_size=0.3,
            random_state=123, shuffle=True, stratify=split_criteria)


    split_criteria = np.where(train_data[:, 48] > 2.5, 1, 0)
    train_data, val_data, train_exist, val_exist, train_label, val_label, train_split, _ = \
        train_test_split(train_data,train_exist,train_label,split_criteria, test_size =0.2/(0.5+0.2),
            random_state=123, shuffle=True, stratify=split_criteria)

    if oversampling in ['borderline_smote', 'svm_smote','smoteenn', 'smotetomek','smote']:
        if oversampling == 'borderline_smote':
            smote = BorderlineSMOTE(random_state=random_state)
        elif oversampling == 'svm_smote':
            smote = SVMSMOTE(random_state=random_state)
        elif oversampling == 'smoteenn':
            smote = SMOTEENN(random_state=random_state)
        elif oversampling == 'smotetomek':
            smote = SMOTETomek(random_state=random_state)
        elif oversampling == 'smote':
            smote = SMOTE(random_state=random_state)


            #    print("train data before shape", train_data.shape, train_label.shape, type(train_data))
        X, y = smote.fit_resample(train_data, train_label.argmax(axis=1))
        y = np.array([[1,0],[0,1]])[y]
        exist = np.ones_like(X)
        train_data, train_exist, train_label = (X, exist, y)
    #   print("train data after shape", train_data.shape, train_label.shape, type(train_data))

    print(train_label.shape[0],np.sum(train_label[:,0]),np.sum(train_label[:, 1]))
    print(val_label.shape[0],np.sum(val_label[:,0]),np.sum(val_label[:, 1]))
    print(test_label.shape[0],np.sum(test_label[:,0]),np.sum(test_label[:, 1]))
    train_pos_weight = train_label.shape[0] / np.sum(train_label[:, 1]) - 1
    val_pos_weight = val_label.shape[0] / np.sum(val_label[:, 1]) - 1
    test_pos_weight = test_label.shape[0] / np.sum(test_label[:, 1]) - 1
    print("train_pos_weight: ",train_pos_weight)
    print("val_pos_weight: ",val_pos_weight)
    print("test_pos_weight: ",test_pos_weight)
    print("train_data/exist/label shape", train_data.shape, train_exist.shape, train_label.shape)
    print("test_data/exist/label shape", test_data.shape, test_exist.shape, test_label.shape)
    print("val_data/exist/label shape", val_data.shape, val_exist.shape, val_label.shape)
    train_set = TensorDataset(torch.from_numpy(train_data),
                torch.from_numpy(train_exist), torch.from_numpy(train_label))

    val_set = TensorDataset(torch.from_numpy(val_data),
                torch.from_numpy(val_exist), torch.from_numpy(val_label))

    test_set = TensorDataset(torch.from_numpy(test_data),
            torch.from_numpy(test_exist), torch.from_numpy(test_label))

    return train_set,val_set, test_set, train_data.shape[1], all_set


