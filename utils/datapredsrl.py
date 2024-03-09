import numpy as np
import pandas as pd
from tqdm import trange
from sktime.performance_metrics.forecasting import \
    mean_absolute_error
import os


def compute_mae_error(y, bm_preds, flag):
    loss_df = pd.DataFrame()
    for i in trange(bm_preds.shape[1], desc=f'[Compute Error {flag}]'):
        model_mae_loss = [mean_absolute_error(y[j], bm_preds[j, i, :], symmetric=True) for j in range(len(y))]
        loss_df[i] = model_mae_loss
    return loss_df

def load_data_rl(root, setting):
    path_ds = os.path.join(root, setting)
    input_data = np.load(f'{path_ds}/input.npz', allow_pickle=True)
    train_X = input_data['train_X']
    valid_X = input_data['valid_X']
    test_X  = input_data['test_X' ]
    train_y  = input_data['train_y' ]
    valid_y  = input_data['valid_y' ]
    test_labels = input_data["test_labels"]
    train_error = input_data['train_error'] 
    valid_error = input_data['valid_error']  
    test_error  = input_data['test_error' ]  
    return (train_X, valid_X, test_X, train_y, valid_y, test_labels, 
            train_error, valid_error, test_error)

def unify_input_data(args, setting):
    path_ds = os.path.join(args.root_path, setting)

    train_X        = np.load(f'{path_ds}/train_X.npy', allow_pickle=True).astype(np.float32)
    valid_X        = np.load(f'{path_ds}/valid_X.npy', allow_pickle=True).astype(np.float32)
    test_X         = np.load(f'{path_ds}/test_X.npy', allow_pickle=True).astype(np.float32)         
    test_labels    = np.load(f'{path_ds}/test_y.npy', allow_pickle=True).astype(np.int32)  

    train_y = train_X.reshape(train_X.shape[0] * train_X.shape[1],-1).astype(np.float32) 
    valid_y = valid_X.reshape(valid_X.shape[0] * valid_X.shape[1],-1).astype(np.float32)  
    test_y  = test_X.reshape(test_X.shape[0] * test_X.shape[1],-1).astype(np.float32)

    L_test, L_train = len(test_labels), len(train_X)
    L = L_test if L_test < L_train else L_train

    train_X = train_X[:L]
    valid_X = valid_X[:L]
    test_X = test_X[:L]

    train_y = train_y[:L]
    valid_y = valid_y[:L]
    test_y = test_y[:L]

    test_labels = test_labels[:L]

    # predictions
    MODEL_LEARNER = [f"learner{i+1}" for i in range(args.n_learner)]

    bm_train_preds = np.load(f'{path_ds}/bm_train_preds.npz', allow_pickle=True)
    bm_valid_preds = np.load(f'{path_ds}/bm_valid_preds.npz', allow_pickle=True)
    bm_test_preds = np.load(f'{path_ds}/bm_test_preds.npz', allow_pickle=True)

    merge_train = [np.expand_dims(bm_train_preds[model_name].reshape(bm_train_preds[model_name].shape[0] * bm_train_preds[model_name].shape[1],-1), axis=1) for model_name in MODEL_LEARNER]
    merge_valid = [np.expand_dims(bm_valid_preds[model_name].reshape(bm_valid_preds[model_name].shape[0] * bm_valid_preds[model_name].shape[1],-1), axis=1) for model_name in MODEL_LEARNER]
    merge_test = [np.expand_dims(bm_test_preds[model_name].reshape(bm_test_preds[model_name].shape[0] * bm_test_preds[model_name].shape[1],-1), axis=1) for model_name in MODEL_LEARNER]

    train_preds = np.concatenate(merge_train, axis=1)[:L].astype(np.float32)
    valid_preds = np.concatenate(merge_valid, axis=1)[:L].astype(np.float32)
    test_preds = np.concatenate(merge_test, axis=1)[:L].astype(np.float32)

    np.save(f'{path_ds}/bm_train_preds_new.npy', train_preds)
    np.save(f'{path_ds}/bm_valid_preds_new.npy', valid_preds)
    np.save(f'{path_ds}/bm_test_preds_new.npy', test_preds)

    train_error_df = compute_mae_error(train_y, train_preds, "TRAIN")
    valid_error_df = compute_mae_error(valid_y, valid_preds, "VALIDATION")
    test_error_df = compute_mae_error(test_y, test_preds, "TEST")

    np.savez(f'{path_ds}/input.npz',
             train_X=train_X,
             valid_X=valid_X,
             test_X=test_X,
             train_y=train_y,
             test_labels = test_labels,
             valid_y=valid_y,
             train_error=train_error_df,
             valid_error=valid_error_df,
             test_error=test_error_df
            )