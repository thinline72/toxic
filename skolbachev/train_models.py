import argparse
from local_utils import *

parser = argparse.ArgumentParser(description='Training KFolds')
parser.add_argument('-m', '--model_name', metavar='model_name')

parser.add_argument('-init_k', '--init_fold', metavar='init_fold', type=int, default=0)
parser.add_argument('-k', '--num_folds', metavar='num_folds', type=int, default=10)
parser.add_argument('-e', '--epochs', metavar='epochs', type=int, default=25)
parser.add_argument('-bs', '--batch_size', metavar='batch_size', type=int, default=256)

def main():
    global args
    args = parser.parse_args()
    
    if args.model_name == "BiCuDNNGRUx2Model":
        get_model_fun = getBiCuDNNGRUx2Model
    
    train_kfold_models(get_model_fun, args.model_name, [args.folder], [args.test_folder], [tuple(args.shape)], 
                       args.init_fold, args.num_folds, args.epochs, args.ps_epochs, 
                       args.batch_size, args.ps_batch_size, args.ps_test_batch_size)
            
if __name__ == '__main__':
    main() 