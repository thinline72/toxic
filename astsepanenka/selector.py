# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 00:36:38 2018

@author: Артур
"""
from scipy.stats import rankdata
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

def corr(first_file, second_file):
    first_df = pd.read_csv(first_file, index_col=0)
    second_df = pd.read_csv(second_file, index_col=0)
    correlations_list = []
    for class_name in class_names:
              correlations_list.append(first_df[class_name].corr(second_df[class_name], method='pearson'))
    mean_correlation = np.array(correlations_list).mean()
    return mean_correlation

def checker(list_of_subms,treshhold=0.98):
    shape = len(list_of_subms)
    selector = np.zeros((shape,shape))
    for first in list_of_subms:
        for second in list_of_subms:
            if corr(first,second) < treshhold:
                selector[list_of_subms.index(first),list_of_subms.index(second)] = 1
            else:
                selector[list_of_subms.index(first),list_of_subms.index(second)] = 0
    plt.imshow(selector, cmap='hot', interpolation='nearest')
    axes = np.arange(0,len(list_of_subms))
    plt.xticks(list(axes),list_of_subms)
    plt.yticks(list(axes),list_of_subms)
    plt.show()
    print(selector)


            
            
    