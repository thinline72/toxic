import argparse
from local_utils import *

parser = argparse.ArgumentParser(description='Predicting KFolds')
parser.add_argument('-model', '--model_name', metavar='model_name')
parser.add_argument('-emb', '--embs_name', metavar='embs_name')
parser.add_argument('-k', '--num_folds', metavar='num_folds', type=int, default=10)
    
# TODO refactor
def predict_kfold_emb_models(model_name, embs_name, num_folds, seed=seed):
    ids, comments, Y, test_ids, test_comments, inx2label, label2inx = load_data()
    
    comments = Parallel(n_jobs=cpu_cores)(delayed(preprocess)(text) for text in comments)
    test_comments = Parallel(n_jobs=cpu_cores)(delayed(preprocess)(text) for text in test_comments)
    
    vectors, inx2word, word2inx = load_embs(embs_name=embs_name)
    text_analyzer = TextAnalyzer(word2inx, vectors, process_oov_words=True, 
                                 oov_min_doc_hits=5, max_len=450, cpu_cores=cpu_cores)
    seq, meta = text_analyzer.fit_on_texts(comments + test_comments)
    X = seq[:len(comments)]
    test_X = seq[len(comments):]

    trn_folds, val_folds = stratified_kfold_sampling(Y, num_folds, seed)
    
    pred = np.zeros((X.shape[0], Y.shape[1]))
    test_pred = np.zeros((num_folds, test_X.shape[0], Y.shape[1]))
    
    
    for f_inx in range(0, num_folds):
        model_file_name = model_name+"_f"+str(f_inx)
        model_file = models_dir+model_file_name+'.h5'
        
        print("Predicting {}".format(model_file_name))

        model = load_model(model_file, compile=True, 
                           custom_objects={'focal_loss':focal_loss, 'roc_auc_loss':roc_auc_loss, 
                                           'AttentionWeightedAverage':AttentionWeightedAverage,
                                           'Attention':Attention})
        
        pred[val_folds[f_inx]] = model.predict(X[val_folds[f_inx]], batch_size=1024, verbose=0)
        test_pred[f_inx] = model.predict(test_X, batch_size=1024, verbose=0)

        losses = compute_losses(Y[val_folds[f_inx]], pred[val_folds[f_inx]], eps=1e-5)
        print("fold: {}, loss: {}".format(f_inx, sum(losses)/len(losses)))
        print("ROC AUC: {}".format(metrics.roc_auc_score(Y[val_folds[f_inx]], pred[val_folds[f_inx]])))
        print()
        
        del model

    np.save(results_dir+model_name+"_pred.npy", pred)
    np.save(results_dir+model_name+"_test_pred.npy", test_pred)

    losses = compute_losses(Y, pred, eps=1e-5)
    print("full loss: {}".format(sum(losses)/len(losses)))
    print("ROC AUC: {}".format(metrics.roc_auc_score(Y, pred)))
    print()
    
def main():
    global args
    args = parser.parse_args()
        
    predict_kfold_emb_models(args.model_name, args.embs_name, args.num_folds)
            
if __name__ == '__main__':
    main()