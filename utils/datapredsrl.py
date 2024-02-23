import numpy as np
import pandas as pd
from tqdm import trange
from sktime.performance_metrics.forecasting import \
    mean_absolute_error

def compute_mae_error(y, bm_preds):
    loss_df = pd.DataFrame()
    for i in trange(bm_preds.shape[1], desc='[Compute Error]'):
        model_mae_loss = [mean_absolute_error(y[j], bm_preds[j, i, :], symmetric=True) for j in range(len(y))]
        loss_df[i] = model_mae_loss
    return loss_df

def load_data_rl(root):
    input_data = np.load(f'{root}input.npz')
    train_X = input_data['train_X']
    valid_X = input_data['valid_X']
    test_X  = input_data['test_X' ]
    test_y  = input_data['test_y' ]
    train_error = input_data['train_error']  # (55928, 9)
    valid_error = input_data['valid_error']  # (6867,  9)
    test_error  = input_data['test_error' ]  # (6867,  9)
    return (train_X, valid_X, test_X, test_y,
            train_error, valid_error, test_error)

def unify_input_data(args):
    train_X   = np.load(f'{args.root_path}train_X.npy')
    valid_X   = np.load(f'{args.root_path}valid_X.npy')
    test_X    = np.load(f'{args.root_path}test_X.npy')         
    test_y    = np.load(f'{args.root_path}test_y.npy')        

    # predictions
    MODEL_LEARNER = [f"learner{i+1}" for i in range(args.n_learner)]
    bm_train_preds = np.load(f'{args.root_path}bm_train_preds.npz')
    bm_valid_preds = np.load(f'{args.root_path}/bm_valid_preds.npz')
    bm_test_preds = np.load(f'{args.root_path}/bm_test_preds.npz')
    merge_train, merge_valid, merge_test = [],[],[]
    for model_name in MODEL_LEARNER:
        model_train_pred = bm_train_preds[model_name]
        model_valid_pred = bm_valid_preds[model_name]
        model_test_pred = bm_test_preds[model_name]

        model_train_pred = np.expand_dims(model_train_pred, axis=1) # menambah 1 dimensi
        model_valid_pred = np.expand_dims(model_valid_pred, axis=1) # menambah 1 dimensi
        model_test_pred = np.expand_dims(model_test_pred, axis=1) # menambah 1 dimensi

        merge_train.append(model_train_pred)
        merge_valid.append(model_valid_pred)
        merge_test.append(model_test_pred)

    train_preds = np.concatenate(merge_train, axis=1)
    valid_preds = np.concatenate(merge_valid, axis=1)
    test_preds = np.concatenate(merge_test, axis=1)

    np.save(f'{args.root_path}bm_train_preds_new.npy', train_preds)
    np.save(f'{args.root_path}bm_valid_preds_new.npy', valid_preds)
    np.save(f'{args.root_path}bm_test_preds.npy', test_preds)

    train_error_df = compute_mae_error(train_X, train_preds)
    valid_error_df = compute_mae_error(valid_X, valid_preds)
    test_error_df  = compute_mae_error(test_y , test_preds)

    np.savez('dataset/input.npz',
             train_X=train_X,
             valid_X=valid_X,
             test_X=test_X,
             test_y=test_y,
             train_error=train_error_df,
             valid_error=valid_error_df,
             test_error=test_error_df
            )
    
