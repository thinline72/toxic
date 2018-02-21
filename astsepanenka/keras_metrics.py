import tensorflow as tf
import keras.backend as K
import numpy as np
from sklearn.metrics import roc_auc_score

#metrics

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

# losses

def u_statistic_loss(y_true,y_pred):
    with tf.name_scope("u_statistic_loss"):
        pos = tf.boolean_mask(y_pred, tf.cast(y_true, tf.bool))
        neg = tf.boolean_mask(y_pred, ~tf.cast(y_true, tf.bool))
        pos = tf.expand_dims(pos, 0)
        neg = tf.expand_dims(neg, 1)
        gamma = 0.2
        p = 3
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
