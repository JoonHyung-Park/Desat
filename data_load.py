import numpy as np
import pandas as pd

import os
import yaml

import torch
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

def normalize(X, exist=None, mean=None, std=None):
    rest_ind = [0, 4, 21, 22, 23, 27, 28, 29, 30, 32, 33, 34, 35, 36, 37, 38,
            39, 40, 41, 42, 43, 44, 45, 46, 55, 59, 60] # exclude ASA, gender
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
        random_state=123, ensemble=False, oversampling='borderline_smote'):
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

    train_data, test_data, train_exist, test_exist, train_label, test_label, train_split, _ = \
        train_test_split(all_data, all_exist, all_label, split_criteria, test_size=0.3,
            random_state=123, shuffle=True, stratify=split_criteria)


    # Split data into k fold for k-fold cross validation
    kf = StratifiedKFold(n_splits, shuffle=True, random_state=random_state) # fixed random state


    kf_gen = kf.split(train_data, train_split) #ASA score

    def k_folds():
        for train_ind, val_ind in kf_gen:
            fold_train = (train_data[train_ind], train_exist[train_ind],
                    train_label[train_ind])
            fold_val = (train_data[val_ind], train_exist[val_ind],
                train_label[val_ind])
            fold_val = TensorDataset(*from_numpy(*fold_val))

            if oversampling in ['borderline_smote', 'svm_smote', 'smotenc']:
                from imblearn.over_sampling import SMOTENC, BorderlineSMOTE, SVMSMOTE
                if oversampling == 'borderline_smote':
                    smote = BorderlineSMOTE(random_state=random_state)
                elif oversampling == 'svm_smote':
                    smote = SVMSMOTE(random_state=random_state)
                elif oversampling == 'smotenc':
                    categorical_features = [2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                            20, 24, 25, 26, 31, 47, 49, 50, 51, 52, 53, 54, 56, 57, 58] + [1, 48]
                    smote = SMOTENC(categorical_features=categorical_features,
                            random_state=random_state, sampling_strategy='auto', neighbors=5,
                            )
                #print(oversampling)
                X, y = smote.fit_resample(fold_train[0], fold_train[2].argmax(axis=1))
                y = np.array([[1, 0],[0, 1]])[y]
                exist = np.ones_like(X)

                fold_train = (X, exist, y)

            pos_weight =  fold_train[2].shape[0] / np.sum(fold_train[2][:, 1]) - 1

            '''
            if ensemble:
                pos_ind, = np.where(fold_train[-1][:, 0] == 0)
                neg_ind, = np.where(fold_train[-1][:, 0] == 1)

                pos_data, neg_data = fold_train[0][pos_ind], fold_train[0][neg_ind]
                pos_exist, neg_exist = fold_train[1][pos_ind], fold_train[1][neg_ind]

                kf = KFold(n_splits=int(pos_weight / 1), shuffle=True,
                            random_state=random_state)

                splited_neg = [(neg_data[ind], neg_exist[ind]) for _, ind in
                        kf.split(neg_data)]
                ensemble_datatemp = \
                    [TensorDataset(*from_numpy(
                      np.concatenate((pos_data, neg_data), axis=0),
                      np.concatenate((pos_exist, neg_exist), axis=0),
                      np.concatenate((np.tile([[0, 1]], (len(pos_data), 1)),
                                      np.tile([[1, 0]], (len(neg_data),
                                          1))),axis=0))) \
                        for neg_data, neg_exist in splited_neg]
                fold_train = ensemble_datatemp
                pos_weight = 1\
                '''
            fold_train = TensorDataset(*from_numpy(*fold_train))

            yield fold_train, fold_val, pos_weight

    print("test_data/exist/label shape", test_data.shape, test_exist.shape, test_label.shape)
    test_set = TensorDataset(torch.from_numpy(test_data),
            torch.from_numpy(test_exist), torch.from_numpy(test_label))
    return k_folds(), test_set, train_data.shape[1], all_set
