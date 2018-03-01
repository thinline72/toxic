import argparse
from local_utils import *

parser = argparse.ArgumentParser(description='Training KFolds')
parser.add_argument('-m_name', '--model_name', metavar='model_name')
parser.add_argument('-m_type', '--model_type', metavar='model_type', type=int)
parser.add_argument('-emb', '--embs_name', metavar='embs_name')
parser.add_argument('-emb_dropout', '--emb_dropout', metavar='emb_dropout', type=float, default=0.5)

parser.add_argument('-init_k', '--init_fold', metavar='init_fold', type=int, default=0)
parser.add_argument('-k', '--num_folds', metavar='num_folds', type=int, default=10)
parser.add_argument('-e', '--epochs', metavar='epochs', type=int, default=24)
parser.add_argument('-ae', '--add_epochs', metavar='add_epochs', type=int, default=8)
parser.add_argument('-str_b', '--stratified_batches', metavar='stratified_batches', type=int, default=0)
parser.add_argument('-bs', '--batch_size', metavar='batch_size', type=int, default=256)
parser.add_argument('-max_len', '--max_len', metavar='max_len', type=int, default=460)

def train_kfold_emb_models(model_fun, model_name, embs_name, emb_dropout, stratified_batches, 
                           init_fold, num_folds, epochs, add_epochs, batch_size, max_len, seed=seed):

    os.makedirs(models_dir+model_name, exist_ok=True)
    
    # Load and preprocess data
    ids, comments, Y, test_ids, test_comments, inx2label, label2inx = load_data()
    Y_wblank = np.concatenate([Y, np.expand_dims((~Y.any(axis=1)).astype(int), 1)], axis=1)
    
    vectors, inx2word, word2inx = load_embs(embs_name=embs_name)
    text_analyzer = TextAnalyzer(word2inx, vectors, max_len=max_len, process_oov_words=True, oov_min_doc_hits=5)

    docs = pickle.load(open('data/tokenized_comments.pkl', 'rb'))
    seq, meta = text_analyzer.fit_on_docs(docs)
    
    np.save(models_dir+model_name+"/emb_vectors.npy", text_analyzer.emb_vectors)
    pickle.dump(text_analyzer.emb2inx, open(models_dir+model_name+'/emb2inx.pkl', 'wb'))
    pickle.dump(text_analyzer.inx2emb, open(models_dir+model_name+'/inx2emb.pkl', 'wb'))
    
    X = seq[:len(comments)]
    test_X = seq[len(comments):]
    np.save(models_dir+model_name+"/X.npy", X)
    np.save(models_dir+model_name+"/test_X.npy", test_X)

    meta_mean = meta.mean(axis=0)
    meta_std = meta.std(axis=0)
    meta = (meta - meta_mean)/meta_std

    print("mean_len: {}".format(meta_mean[0]))
    print("mean_len + 2*std: {}".format(meta_mean[0]+2*meta_std[0]))
    print("mean_len + 3*std: {}".format(meta_mean[0]+3*meta_std[0]))

    X_meta = meta[:len(comments)]
    test_X_meta = meta[len(comments):]
    np.save(models_dir+model_name+"/X_meta.npy", X_meta)
    np.save(models_dir+model_name+"/test_X_meta.npy", test_X_meta)
    

    # Splitting
    trn_folds, val_folds = stratified_kfold_sampling(Y, num_folds, seed)
    pred = np.zeros((X.shape[0], Y.shape[1]))
    test_pred = np.zeros((num_folds, test_X.shape[0], Y.shape[1]))
    
    
    # Training
    for f_inx in range(init_fold,num_folds):
        print("Training fold {}".format(f_inx))
        
        model = model_fun(input_shape=X.shape[1], classes=Y.shape[1], num_words=len(text_analyzer.inx2emb), 
                          emb_size=text_analyzer.emb_size, emb_matrix=text_analyzer.emb_vectors, emb_dropout=emb_dropout,
                          attention=0, dense=False, emb_trainable=False)
        model_file = models_dir+model_name+"/fold_"+str(f_inx)+".h5"

        # Train and valid seq
        if stratified_batches: trn_seq = StratifiedFeatureSequence(X[trn_folds[f_inx]], Y[trn_folds[f_inx]], batch_size)
        else: trn_seq = FeatureSequence(X[trn_folds[f_inx]], X_meta[trn_folds[f_inx]], Y[trn_folds[f_inx]], 
                                        batch_size, shuffle=True)
        val_seq = FeatureSequence(X[val_folds[f_inx]], X_meta[val_folds[f_inx]], Y[val_folds[f_inx]], 1024)       

        # Callbacks
        model_checkpoint = ModelCheckpoint(model_file, monitor='val_loss', mode='min', verbose=1, save_best_only=True)
        early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=6, verbose=1, mode='auto')
        roc_auc_eval = RocAucEvaluation(X[val_folds[f_inx]], Y[val_folds[f_inx]], batch_size=1024)
        clr = CyclicLR(base_lr=0.0001, max_lr=0.003, step_size=2*len(trn_seq), mode='triangular2')
        
        print("Training "+model_name+" ...")
        model.compile(loss="binary_crossentropy", optimizer=optimizers.Nadam())
        model.fit_generator(
            generator=trn_seq, steps_per_epoch=len(trn_seq),
            validation_data=val_seq, validation_steps=len(val_seq),
            initial_epoch=0, epochs=epochs, shuffle=False, verbose=2,
            callbacks=[model_checkpoint, clr, early_stop, roc_auc_eval],
            use_multiprocessing=False, workers=cpu_cores, max_queue_size=8*cpu_cores)
        
        del model
        model = load_model(model_file, compile=True, 
                           custom_objects={'Attention':Attention, 'art_loss':art_loss,
                                           'AttentionWeightedAverage':AttentionWeightedAverage})
            
        clr = CyclicLR(base_lr=0.0001, max_lr=0.001, step_size=2*len(trn_seq), mode='triangular2')
        early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=4, verbose=1, mode='auto')
        model.compile(loss="binary_crossentropy", optimizer=optimizers.Adam())
        model.fit_generator(
            generator=trn_seq, steps_per_epoch=len(trn_seq),
            validation_data=val_seq, validation_steps=len(val_seq),
            initial_epoch=epochs, epochs=epochs+add_epochs, shuffle=False, verbose=1,
            callbacks=[model_checkpoint, clr, early_stop, roc_auc_eval],
            use_multiprocessing=False, workers=cpu_cores, max_queue_size=8*cpu_cores)
        
        
        print("Predicting ...")
        del model
        model = load_model(model_file, compile=True, 
                           custom_objects={'Attention':Attention, 'art_loss':art_loss,
                                           'AttentionWeightedAverage':AttentionWeightedAverage})
        
        pred[val_folds[f_inx]] = model.predict(X[val_folds[f_inx]], batch_size=1024, verbose=0)
        test_pred[f_inx] = model.predict(test_X, batch_size=1024, verbose=0)

        losses = compute_losses(Y[val_folds[f_inx]], pred[val_folds[f_inx]], eps=1e-5)
        print("fold: {}, loss: {}".format(f_inx, sum(losses)/len(losses)))
        print("ROC AUC: {}".format(metrics.roc_auc_score(Y[val_folds[f_inx]], pred[val_folds[f_inx]])))
        print()

    np.save(models_dir+model_name+"/X_pred.npy", pred)
    np.save(models_dir+model_name+"/test_X_pred.npy", test_pred)

    losses = compute_losses(Y, pred, eps=1e-5)
    print("full loss: {}".format(sum(losses)/len(losses)))
    print("ROC AUC: {}".format(metrics.roc_auc_score(Y, pred)))
    print()
    
def main():
    global args
    args = parser.parse_args()
    
    print("Model type: {}".format(args.model_type))
    if args.model_type == 0:
        model_fun = getModel0
    if args.model_type == 1:
        model_fun = getModel1
    if args.model_type == 2:
        model_fun = getModel2
    if args.model_type == 3:
        model_fun = getModel3

    train_kfold_emb_models(model_fun, args.model_name, args.embs_name, args.emb_dropout,
                           True if args.stratified_batches==1 else False,
                           args.init_fold, args.num_folds, args.epochs, args.add_epochs, args.batch_size, args.max_len)
            
if __name__ == '__main__':
    main()