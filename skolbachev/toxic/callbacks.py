from __future__ import absolute_import

from keras.callbacks import *
from sklearn import metrics as metrics

class ROCAUCCallback(Callback):
    def __init__(self, X_val, Y_val, batch_size):
        super(ROCAUCCallback, self).__init__()
        
        self.X_val = X_val
        self.Y_val = Y_val
        self.batch_size = batch_size
 
    def on_epoch_end(self, epoch, logs={}):           
        Y_val_pred = self.model.predict(self.X_val, batch_size=self.batch_size)
        score = metrics.roc_auc_score(self.Y_val, Y_val_pred)
        print('\rroc-auc_val: %s' % (str(round(score,8))), end=100*' '+'\n')
        return