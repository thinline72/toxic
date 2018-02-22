# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 00:36:38 2018

@author: Arthur Stsepanenka
"""
from scipy.stats import rankdata
import numpy as np
import pandas as pd

def blending_auc(list_of_arrays,list_of_weights):
    final = rankdata(np.copy(list_of_arrays[0]))*list_of_weights[0]
    for element in range(1,len(list_of_weights)-1):
        final = final + list_of_weights[element]*rankdata(list_of_arrays[element])
    return final
    
def blending_submissions(list_of_subm,list_of_weights,filename='blending.csv'):
    final = list_of_subm[0].copy()
    label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    for label in label_cols:
        list_of_arrays = []
        for subm in list_of_subm:
            list_of_arrays.append(subm[label])
        final_label = blending_auc(list_of_arrays,list_of_weights)
        final[label] = final_label
        final.to_csv(filename,index=False)

            
            
    
