from os import walk
import pandas as pd
import numpy as np

foldn = '9'

cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

f = []
for (dirpath, dirnames, filenames) in walk('../blend/'):
    f.extend(filenames)
    break

ff = []
for (dirpath, dirnames, filenames) in walk('../skolbachev/'):
    for d in dirnames:
        for (dirpath, dirnames2, filenames) in walk('../skolbachev/'+d):
            for qf in filenames:
                ff.append('../skolbachev/'+d+'/'+qf)
                
chen_sol = pd.read_csv('../cheng/ensemble/'+foldn+'/gru.info.dsfu.ngram.dsfu.lower_model.ckpt-20.00-44860.valid').sort_values('id').reset_index(drop=True)
chen_sol_ids = chen_sol['id'].values

fchen = []
for (dirpath, dirnames, filenames) in walk('../cheng/ensemble/'+foldn):
    fchen.extend([q for q in filenames if q.endswith('.valid')])
    break 

train_idx = pd.read_csv('../input/train.csv')['id'].values
train = pd.read_csv('../input/train.csv').sort_values('id').reset_index(drop=True)

from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import minmax_scale

train = train.loc[train['id'].isin(chen_sol_ids),:].sort_values('id').reset_index(drop=True)

oofs = []
onms = []

train_files = [q for q in f if q.startswith('train')]
for q in train_files:
    nm = q[6:-4]
    nf = pd.read_csv('../blend/'+q)
    if 'fold_id' in nf.columns:
        nf = nf.drop(['fold_id'],axis=1)
    nf = nf.loc[nf.id.isin(chen_sol_ids),:].sort_values('id').reset_index(drop=True)
    for c in cols:
        if 'identity_hate' in nf.columns:
            nf[c] = minmax_scale(nf[c])
        else:
            nf[c] = minmax_scale(nf[c+'_oof'])
            nf.drop([c+'_oof'],axis=1,inplace=True)
        #print(nm,c,roc_auc_score(train[c],nf[c]))
    if (nf.columns.tolist().index('id')==0):
        nf.columns = ['id'] + [nm+'_' + q for q in cols]
    else:
        nf.columns = [nm+'_' + q for q in cols] + ['id']
    print(nm, roc_auc_score(train[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult','identity_hate']],nf[[nm+'_toxic',nm+'_severe_toxic',nm+'_obscene',nm+'_threat',nm+'_insult',nm+'_identity_hate']]))
    onms.append(nm)
    oofs.append(nf)
    
sk_train = [q for q in ff if not q.endswith('test_X_pred.npy')]
suf = 'sk'
i = 0
for q in sk_train:
    nf = pd.DataFrame(np.load(q))
    nm = suf+str(i)
    nf.columns = [nm+'_'+q for q in cols]
    nf['id'] = train_idx
    nf = nf.loc[nf.id.isin(chen_sol_ids),:].sort_values('id').reset_index(drop=True)
    for c in cols:
        nf[nm+'_'+c] = minmax_scale(nf[nm+'_'+c])
    print(nm, roc_auc_score(train[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult','identity_hate']],nf[[nm+'_toxic',nm+'_severe_toxic',nm+'_obscene',nm+'_threat',nm+'_insult',nm+'_identity_hate']]))
    onms.append(nm)
    oofs.append(nf)
    i = i + 1
    
suf = 'chen'
i = 0
for q in fchen:
    nf = pd.read_csv('../cheng/ensemble/'+foldn+'/'+q)
    nm = suf+str(i)
    nf.columns = ['id'] + [nm+'_'+q for q in cols]
    nf = nf.sort_values('id').reset_index(drop=True)
    for c in cols:
        nf[nm+'_'+c] = minmax_scale(nf[nm+'_'+c])
    try:
        print(nm, roc_auc_score(
                    train[cols],
                    nf[[nm+'_toxic',nm+'_severe_toxic',nm+'_obscene',nm+'_threat',nm+'_insult',nm+'_identity_hate']]))
        onms.append(nm)
        oofs.append(nf)
    except:
        nf = nf.loc[nf['id'] != '0',:].reset_index(drop=True)
        print(nm, roc_auc_score(
                    train[cols],
                    nf[[nm+'_toxic',nm+'_severe_toxic',nm+'_obscene',nm+'_threat',nm+'_insult',nm+'_identity_hate']]))
        onms.append(nm)
        oofs.append(nf)
        pass
    i = i + 1  

train = pd.read_csv('../input/train.csv')
train = train.loc[train['id'].isin(chen_sol_ids),:].sort_values('id').reset_index(drop=True)
for o in oofs:
    train = train.merge(o, on='id', how='left')

orig_cols = ['id', 'comment_text', 'toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

print(len(onms))        

def evaluate_nms(nms,c):
    scores = {}
    y = train[c]
    scores[c] = []
    for n in nms:
        w = roc_auc_score(y,train[n+'_'+c])
        scores[c].append(w)        
    p = []
    ws = scores[c]
    y = train[c]
    pred = 0
    i = 0
    for n in nms:
        pred += ((ws[i]-np.min(ws))/(np.max(ws)-np.min(ws))+0.01)*minmax_scale(train[n+'_'+c])
        i = i + 1
    p.append(roc_auc_score(y,pred))
    return np.mean(p)

best_sets = {}
for c in cols:
    bst = evaluate_nms(onms,c)
    best_set = onms
    while True:
        d = {}
        bst_j_remove = ''
        bst_j_add = ''
        for j in best_set:
            nms = list(set(best_set) - set([j]))
            d[j] = evaluate_nms(nms,c)
            if d[j] >= bst:
                bst = d[j]
                bst_j_remove = j
        for j in onms:
            if j in best_set:
                continue
            nms = list(set(best_set) | set([j]))
            d[j] = evaluate_nms(nms,c)
            if d[j] > bst:
                bst_j_remove = ''
                bst = d[j]
                bst_j_add = j
        if bst_j_remove == '' and bst_j_add == '':
            break
        if bst_j_remove != '':
            best_set = list(set(best_set) - set([bst_j_remove]))
        else:
            best_set = list(set(best_set) | set([bst_j_add]))
        print(c,bst,best_set)
    best_sets[c] = best_set.copy()    
    
from scipy.optimize import minimize
def fns(x,c):
    y = train[c]
    pred = 0
    i = 0
    for n in best_sets[c]:
        pred += x[i]*(minmax_scale(train[n+'_'+c]))
        i = i + 1
    return -roc_auc_score(y,pred)

bweights = {}
p = []
for c in cols:
    y = train[c]
    ws = []
    for n in best_sets[c]:
        w = roc_auc_score(y,train[n+'_'+c])
        ws.append(w)     
    i = 0
    weights = []
    for n in best_sets[c]:
        weights.append(((ws[i]-np.min(ws))/(np.max(ws)-np.min(ws))+0.01)*1e-5)
        i = i + 1
    res = minimize(fns, weights, args = c, method='Nelder-Mead', tol=1e-8)
    bweights[c] = res.x / np.sum(res.x)
    p.append(-fns(bweights[c],c))
    #print(c,p[-1],bweights[c])
print(p,'\t',np.mean(p))

preds = []

train_files = [q for q in f if q.startswith('test')]
for q in train_files:
    nf = pd.read_csv('../blend/'+q)
    if 'fold_id' in nf.columns:
        ssc = preds[0].copy()
        for c in cols:
            ssc[c] = 0
        for c in cols:
            qq = nf[['id',c]].groupby(['id']).agg('mean').reset_index().sort_values('id').reset_index(drop=True)[c]
            ssc[c] = minmax_scale(qq.values)
        nf = ssc
    for c in cols:
        nf[c] = minmax_scale(nf[c])
    preds.append(nf)
    
sk_train = [q for q in ff if q.endswith('test_X_pred.npy')]
suf = 'sk'
i = 0
for q in sk_train:
    nf = pd.DataFrame(np.mean(np.load(q),axis=0))
    nf.columns = cols
    for c in cols:
        nf[c] = minmax_scale(nf[c])
    preds.append(nf)
    i = i + 1

suf = 'chen'
i = 0
for q in fchen:
    nf = pd.read_csv('../cheng/ensemble/'+foldn+'/'+q.replace('.valid','.infer')).sort_values('id').reset_index(drop=True)
    nm = suf+str(i)
    if nm not in onms:
        print(nm)
        i = i + 1
        continue
    for c in cols:
        nf[c] = minmax_scale(nf[c])
    preds.append(nf)
    i = i + 1  

print(len(preds))

sub = pd.read_csv('../input/sample_submission.csv')
for c in cols:
    sub[c] = 0
    y = train[c]
    ws = []
    for n in best_sets[c]:
        w = roc_auc_score(y,train[n+'_'+c])
        ws.append(w)
    k = 0
    for n in best_sets[c]:
        j = onms.index(n)
        sub[c] += bweights[c][k] * preds[j][c]
        k = k + 1
    sub[c] = minmax_scale(sub[c])
sub.head(n=3)

sub.to_csv('weighted_blend_82models_'+foldn+'.csv', index=False)