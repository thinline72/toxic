from __future__ import absolute_import

import numpy as np
import tensorflow as tf
from keras import backend as K
from sklearn.metrics import roc_auc_score

def mean_column_wise_auc(y_true, y_pred):
    assert y_true.shape[1] == y_pred.shape[1],'Arrays must have the same dimension'
    list_of_aucs = []
    for column in range(y_true.shape[1]):
        list_of_aucs.append(roc_auc_score(y_true[:,column],y_pred[:,column]))
    return np.array(list_of_aucs).mean()

def streaming_auc_keras(y_true, y_pred):
    score, up_opt = tf.contrib.metrics.streaming_auc(y_pred, y_true)
    K.get_session().run(tf.local_variables_initializer())
    with tf.control_dependencies([up_opt]):
        score = tf.identity(score)
    return score

def auc_keras(y_true, y_pred):
    score, up_opt = tf.metrics.auc(y_true, y_pred)
    K.get_session().run(tf.local_variables_initializer())
    with tf.control_dependencies([up_opt]):
        score = tf.identity(score)
    return score