import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys


def save_log(message, log):
    print(message)
    with open(log, "a") as log_file:
        log_file.write('\n' + message)

class L2Norm(nn.Module):
    def forward(self, input, dim=1):
        return F.normalize(input, p=2, dim=dim)

def KFold(n=6000, n_folds=10, shuffle=False):
    folds = []
    base = list(range(n))
    for i in range(n_folds):
        test = base[int(i*n/n_folds):int((i+1)*n/n_folds)]
        train = list(set(base)-set(test))
        folds.append([train,test])
    return folds


def eval_acc(threshold, diff):
    y_predict = np.int32(diff[:,0]>threshold)
    y_true = np.int32(diff[:,1])
    accuracy = 1.0*np.count_nonzero(y_true==y_predict)/len(y_true)
    return accuracy


def find_best_threshold(thresholds, predicts):
    best_threshold = best_acc = 0
    for threshold in thresholds:
        accuracy = eval_acc(threshold, predicts)
        if accuracy >= best_acc:
            best_acc = accuracy
            best_threshold = threshold
    return best_threshold


def tensor_pair_cosine_distance(features11, features12, features21, features22, type='normal'):
    if type == 'concat':
        features1 = torch.cat((features11, features12), dim=1)
        features2 = torch.cat((features21, features22), dim=1)
    elif type == 'sum':
        features1 = features11 + features12
        features2 = features21 + features22
    elif type == 'normal':
        features1 = features11
        features2 = features21
    else:
        print('tensor_pair_cosine_distance unspported type!')
        sys.exit()
    scores = torch.nn.CosineSimilarity()(features1, features2)
    scores = scores.cpu().numpy().reshape(-1, 1)
    return scores
