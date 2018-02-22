from __future__ import absolute_import

import numpy as np
import tensorflow as tf
from keras import backend as K

def getClassWeights(Y, mu=0.5):
    return np.array([w for w in np.log(mu*Y.shape[0]/Y.sum(axis=0))])

def focal_loss(y_true, y_pred, alpha, gamma=0.5):
    alpha = K.variable(alpha)
    pt = K.abs(1. - y_true - y_pred)
    pt = K.clip(pt, K.epsilon(), 1. - K.epsilon())
    return K.mean(-alpha * K.pow(1. - pt, gamma) * K.log(pt), axis=-1)

def u_statistic_loss(y_true, y_pred, gamma=0.2, p=3.0):
    """ U statistic loss
    Approximates the Area Under Curve score, using approximation based on
    the Wilcoxon-Mann-Whitney U statistic.
    Yan, L., Dodier, R., Mozer, M. C., & Wolniewicz, R. (2003).
    Optimizing Classifier Performance via an Approximation to the Wilcoxon-Mann-Whitney Statistic.
    Measures overall performance for a full range of threshold levels.
    Arguments:
        y_pred: `Tensor`. Predicted values.
        y_true: `Tensor` . Targets (labels), a probability distribution.
    """
    with tf.name_scope("u_statistic_loss"):
        
        pos = tf.boolean_mask(y_pred, tf.cast(y_true, tf.bool))
        neg = tf.boolean_mask(y_pred, ~tf.cast(y_true, tf.bool))
        
        pos = tf.expand_dims(pos, 0)
        neg = tf.expand_dims(neg, 1)
        
        difference = tf.zeros_like(pos * neg) + pos - neg - gamma
        masked = tf.boolean_mask(difference, difference < 0.0)
        return tf.reduce_sum(tf.pow(-masked, p))

def SoftAUC_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.int32)
    parts = tf.dynamic_partition(y_pred, y_true, 2)
    y_pos = parts[1]
    y_neg = parts[0]
    y_pos = tf.expand_dims(y_pos, 0)
    y_neg = tf.expand_dims(y_neg, -1)
    return K.mean(K.sigmoid(y_neg - y_pos))

def SVMrank_loss(y_true, y_pred):
    margin = 1.0
    y_true = tf.cast(y_true, tf.int32)
    parts = tf.dynamic_partition(y_pred, y_true, 2)
    y_pos = parts[1]
    y_neg = parts[0]
    y_pos = tf.expand_dims(y_pos, 0)
    y_neg = tf.expand_dims(y_neg, -1)
    return K.mean(K.relu(margin - y_neg - y_pos))


###########experimental losses##############

def exp_loss(y_true, y_pred):
    loss = u_statistic_loss(y_true,y_pred) + SoftAUC_loss(y_true, y_pred)
    return loss

def art_loss(y_true, y_pred):
    loss = u_statistic_loss(y_true,y_pred) + SVMrank_loss(y_true, y_pred)
    return loss