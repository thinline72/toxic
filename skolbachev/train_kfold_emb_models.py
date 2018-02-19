import argparse
from local_utils import *

parser = argparse.ArgumentParser(description='Training KFolds')
parser.add_argument('-m_name', '--model_name', metavar='model_name')
parser.add_argument('-m_type', '--model_type', metavar='model_type', type=int, default=0)
parser.add_argument('-emb', '--embs_name', metavar='embs_name')

parser.add_argument('-init_k', '--init_fold', metavar='init_fold', type=int, default=0)
parser.add_argument('-k', '--num_folds', metavar='num_folds', type=int, default=10)
parser.add_argument('-e', '--epochs', metavar='epochs', type=int, default=30)
parser.add_argument('-str_b', '--stratified_batches', metavar='stratified_batches', type=int, default=0)
parser.add_argument('-bs', '--batch_size', metavar='batch_size', type=int, default=256)

def train_kfold_emb_models(model_fun, model_name, embs_name, stratified_batches, 
                           init_fold, num_folds, epochs, batch_size, seed=seed):
    ids, comments, Y, test_ids, test_comments, inx2label, label2inx = load_data()
    
    comments = Parallel(n_jobs=cpu_cores)(delayed(preprocess)(text) for text in comments)
    test_comments = Parallel(n_jobs=cpu_cores)(delayed(preprocess)(text) for text in test_comments)
    
    vectors, inx2word, word2inx = load_embs(embs_name=embs_name)
    text_analyzer = TextAnalyzer(word2inx, vectors, process_oov_words=True, 
                                 oov_min_doc_hits=5, max_len=450, cpu_cores=cpu_cores)
    seq, meta = text_analyzer.fit_on_texts(comments + test_comments)
    
    pickle.dump(text_analyzer.emb_vectors, open(models_dir+model_name+'_emb_vectors.pkl', 'wb'))
    pickle.dump(text_analyzer.emb2inx, open(models_dir+model_name+'_emb2inx.pkl', 'wb'))
    pickle.dump(text_analyzer.inx2emb, open(models_dir+model_name+'_inx2emb.pkl', 'wb'))
    
    X = seq[:len(comments)]
    test_X = seq[len(comments):]

    meta_mean = meta.mean(axis=0)
    meta_std = meta.std(axis=0)
    meta = (meta - meta_mean)/meta_std
    X_meta = meta[:len(comments)]
    test_X_meta = meta[len(comments):]
    print("mean_len + 2*std: {}".format(meta_mean[0]+2*meta_std[0]))
    print("mean_len + 3*std: {}".format(meta_mean[0]+3*meta_std[0]))
    
    
    trn_folds, val_folds = stratified_kfold_sampling(Y, num_folds, seed)
    
    pred = np.zeros((X.shape[0], Y.shape[1]))
    test_pred = np.zeros((num_folds, test_X.shape[0], Y.shape[1]))
    
    
    for f_inx in range(init_fold,num_folds):
        print("Training fold {}".format(f_inx))
        
        model = model_fun(input_shape=X.shape[1], classes=Y.shape[1], num_words=len(text_analyzer.inx2emb), 
                          emb_size=text_analyzer.emb_size, emb_matrix=text_analyzer.emb_vectors,
                          attention=0, dense=False, emb_trainable=False)
        model_file_name = model_name+"_f"+str(f_inx)
        model_file = models_dir+model_file_name+'.h5'

        # Train and valid seq
        if stratified_batches: trn_seq = StratifiedFeatureSequence(X[trn_folds[f_inx]], Y[trn_folds[f_inx]], batch_size)
        else: trn_seq = FeatureSequence(X[trn_folds[f_inx]], X_meta[trn_folds[f_inx]], Y[trn_folds[f_inx]], 
                                        batch_size, shuffle=True)
        val_seq = FeatureSequence(X[val_folds[f_inx]], X_meta[val_folds[f_inx]], Y[val_folds[f_inx]], batch_size)       

        # Callbacks
        model_checkpoint = ModelCheckpoint(model_file, monitor='val_loss', mode='min', verbose=1, save_best_only=True)
        lr_schedule = LearningRateScheduler(lr_change, verbose=1)
        lr_reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=1, min_lr=0.0001, verbose=1)
        early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=4, verbose=1, mode='auto')
        roc_auc_callback = ROCAUCCallback(X[val_folds[f_inx]], Y[val_folds[f_inx]], 1024)

        print("Training "+model_name+" using "+embs_name+" embeddings...")
        weights = getClassWeights(Y)
        model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(0.001, clipvalue=1., clipnorm=1.))
        model.fit_generator(
            generator=trn_seq, steps_per_epoch=len(trn_seq),
            validation_data=val_seq, validation_steps=len(val_seq),
            initial_epoch=0, epochs=epochs, shuffle=False, verbose=2,
            class_weight=weights,
            callbacks=[model_checkpoint, roc_auc_callback, early_stop, lr_reduce],
            use_multiprocessing=False, workers=cpu_cores, max_queue_size=8*cpu_cores)
            
        print("Predicting...")
        del model
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

    np.save(results_dir+model_name+"_pred.npy", pred)
    np.save(results_dir+model_name+"_test_pred.npy", test_pred)

    losses = compute_losses(Y, pred, eps=1e-5)
    print("full loss: {}".format(sum(losses)/len(losses)))
    print("ROC AUC: {}".format(metrics.roc_auc_score(Y, pred)))
    print()
    
def main():
    global args
    args = parser.parse_args()
    
    if args.model_type == 0:
        model_fun = getBiCuDNNGRUx2Model
        print("Model type: BiCuDNNGRUx2")
    elif args.model_type == 1:
        model_fun = getHowardBiCuDNNGRUModel
        print("Model type: HowardBiCuDNNGRU")
        
    train_kfold_emb_models(model_fun, args.model_name, args.embs_name, 
                           True if args.stratified_batches==1 else False,
                           args.init_fold, args.num_folds, args.epochs, args.batch_size)
            
if __name__ == '__main__':
    main()