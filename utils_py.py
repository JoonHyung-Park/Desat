from sklearn.metrics import roc_auc_score
import numpy as np

def get_auc(logits, labels):
    '''
    logits : tensor
    labels : ndarray?
    '''
    score = logits.detach().sigmoid().cpu().squeeze().numpy()
    labels = labels.detach().cpu().numpy()
    return roc_auc_score(labels, score, average = 'samples')
def get_auc_softmax(logits, labels):
    '''
    logits : tensor
    labels : ndarray?
    '''
    score = logits.detach().sigmoid().cpu().squeeze().numpy()
    labels = labels.detach().cpu().numpy()
    return roc_auc_score(labels, score, average = 'samples')

def get_auc_list(logits_l, labels):
    '''
    logits : tensor
    labels : ndarray?
    '''
    score_l = [logits.detach().sigmoid().cpu().squeeze().numpy()
             for logits in logits_l]
    score = np.stack(score_l, axis=0).mean(axis=0)
    return roc_auc_score(labels.detach().cpu(), score)

def get_voting_result(logits_l, labels, thrs=0.5):
    voting_result = np.mean([np.where(
        logits.detach().sigmoid().cpu().squeeze().numpy() < thrs, 0, 1)
            for logits in logits_l], axis=0)
    return roc_auc_score(labels, voting_result)

def choose_and_cut(X, exist=None, cut=True):
    chosen_ind = [0, 23, 48, 49, 42, 20, 53, 54] # Original 8 features
    #chosen_ind = [23, 48, 49, 42, 53, 54, 15, 27, 34, 6, 29, 40, 41, 36, 43, 44, 21] # 17 features
    corresponding_cut = [74, 25, 2.5, 0.5, 30, None, None, None] # 8 features
    #corresponding_cut = [25, 2.5, 0.5, 30, None, None, None, None, None, None, None,None, None,None, None,None, None,] # 17 features
    X = np.copy(X[:, chosen_ind])
    #X = np.copy(X[:,:])
    if cut:
        #X[:, :5] = np.where(X[:, :5] >= np.array(corresponding_cut[:5]).reshape(1, -1), 1, 0)
        X[:, :5] = np.where(X[:, :5] >= np.array(corresponding_cut[:5]).reshape(1, -1), 1, 0)
    if exist is None:
        exist = np.ones_like(X)
    else:
        exist = exist[:, chosen_ind]
    return X * exist, exist
