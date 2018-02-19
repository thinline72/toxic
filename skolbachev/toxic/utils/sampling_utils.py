from __future__ import absolute_import

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, StratifiedKFold

def stratified_sampling(Y, split, random_state):
    train_inx = []
    valid_inx = []
    
    n_classes = Y.shape[1]
    inx = np.arange(Y.shape[0])

    for i in range(0,n_classes):
        sss = StratifiedShuffleSplit(n_splits=1, test_size=split, random_state=random_state+i)
        b_train_inx, b_valid_inx = next(sss.split(inx, Y[:,i]))
        # to ensure there is no repetetion within each split and between the splits
        train_inx = train_inx + list(set(list(b_train_inx)) - set(train_inx) - set(valid_inx))
        valid_inx = valid_inx + list(set(list(b_valid_inx)) - set(train_inx) - set(valid_inx))
        
    return np.array(train_inx), np.array(valid_inx)

def stratified_kfold_sampling(Y, n_splits, random_state):
    train_folds = [[] for _ in range(n_splits)]
    valid_folds = [[] for _ in range(n_splits)]

    n_classes = Y.shape[1]
    inx = np.arange(Y.shape[0])
    valid_size = 1.0 / n_splits

    for cl in range(0, n_classes):
        sss = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state+cl)
        
        for fold, (train_index, test_index) in enumerate(sss.split(inx, Y[:,cl])):
            b_train_inx, b_valid_inx = inx[train_index], inx[test_index]
            
            # to ensure there is no repetetion within each split and between the splits
            train_folds[fold] = train_folds[fold] + list(set(list(b_train_inx)) - set(train_folds[fold]) - set(valid_folds[fold]))
            valid_folds[fold] = valid_folds[fold] + list(set(list(b_valid_inx)) - set(train_folds[fold]) - set(valid_folds[fold]))
        
    return np.array(train_folds), np.array(valid_folds)

def plot_stratified_sampling(Y, train_inx, valid_inx, labels_names, height=12, width=14):
    n_classes = Y.shape[1]
    train_count = []
    valid_count = []
    dist = []
    
    #checking distribution for each class
    for i in range(0, n_classes):
        trn_uniq = np.unique(Y[train_inx,i],return_counts=True)
        if 1.0 in trn_uniq[0]:
            train_count.append(trn_uniq[1][1])
        else:
            train_count.append(0)

        val_uniq = np.unique(Y[valid_inx,i],return_counts=True)
        if 1.0 in val_uniq[0]:
            valid_count.append(val_uniq[1][1])
        else:
            valid_count.append(0)

        dist.append(train_count[-1]/len(train_inx))
        dist.append(valid_count[-1]/len(valid_inx))

    dist_labels = [x for pair in zip([x + '_trn' for x in labels_names],
                                     [x + '_val' for x in labels_names]) for x in pair]

    fig, ax = plt.subplots()
    fig.set_figheight(height)
    fig.set_figwidth(width)
    ax.barh(np.arange(len(dist)), dist)
    ax.set_yticks(np.arange(len(dist)))
    ax.set_yticklabels(dist_labels)
    ax.invert_yaxis()
    ax.set_xlabel('%')
    ax.grid()

    for i, v in enumerate(zip(dist, list(itertools.chain(*zip(train_count, valid_count))))):
        ax.text(v[0] + 0.001, i + 0.25, str(v[1]))

    plt.show()

def plot_stratified_kfold_sampling(Y, train_folds, valid_folds, labels_names, height=12, width=14):
    for train_inx, valid_inx in zip(train_folds, valid_folds):
        plot_stratified_sampling(Y, train_inx, valid_inx, labels_names, height, width)