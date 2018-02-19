from __future__ import absolute_import

import numpy as np
from joblib import Parallel, delayed
from sklearn import metrics as metrics

def find_opt_clip(y_true, y_pred, cl_inx, eps=1e-15, 
                  min_grid=np.arange(0.00001, 0.0001, 0.00001),
                  max_grid=np.arange(0.95, 0.9999, 0.0001)):
    
    opt_min_clip = 0.00001
    opt_max_clip = 0.95  
    
    max_loss = metrics.log_loss(y_true, clip(y_pred, opt_min_clip, opt_max_clip), labels=[0, 1], eps=eps)
    for min_clip in min_grid:
        loss = metrics.log_loss(y_true, clip(y_pred, min_clip, opt_max_clip), labels=[0, 1], eps=eps)
        if loss < max_loss:
            max_loss = loss
            opt_min_clip = min_clip
       
    max_loss = metrics.log_loss(y_true, clip(y_pred, opt_min_clip, opt_max_clip), labels=[0, 1], eps=eps)
    for max_clip in max_grid:
        loss = metrics.log_loss(y_true, clip(y_pred, opt_min_clip, max_clip), labels=[0, 1], eps=eps)
        if loss < max_loss:
            max_loss = loss
            opt_max_clip = max_clip
    
    print("{}: loss {} with {},{}".format(cl_inx, max_loss, opt_min_clip, opt_max_clip))
    return (cl_inx, [opt_min_clip, opt_max_clip])

def find_opt_clip_map(Y_true, Y_pred, cpu_cores=8):
    clips_map = dict(Parallel(n_jobs=cpu_cores)(delayed(find_opt_clip)(Y_true[:,cl_inx], Y_pred[:,cl_inx], cl_inx) 
                                                for cl_inx in np.arange(0, Y_true.shape[1])))
    return [clips_map[i] for i in range(0,Y_true.shape[1])]

def clip(arr, min_value=0.001, max_value=0.95):
    return np.clip(arr, min_value, max_value)