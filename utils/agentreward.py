import torch
import numpy as np
from collections import Counter
from sktime.performance_metrics.forecasting import \
    mean_absolute_error, mean_absolute_percentage_error
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from utils.tools import adjustment

def get_mape_reward(q_mape, mape, R=1):
        q = 0
        while (q < 9) and (mape > q_mape):
            q += 1
        reward = -R + 2*R*(9 - q)/9
        return reward

def get_mae_reward(q_mae, mae, R=1):
    q = 0
    while (q < 9) and (mae > q_mae[q]):
        q += 1
    reward = -R + 2*R*(9 - q)/9
    return reward
# RANK dari
def get_rank_reward(rank, R=1):
        reward = -R + 2*R*(9 - rank)/9
        return reward

def get_batch_rewards(env, idxes, actions, q_mae):
    rewards = []
    # mae_lst = []
    for i in range(len(idxes)):
        rank, new_mae = env.reward_func(idxes[i], actions[i])
        rank_reward = get_rank_reward(rank, 1)
        # mape_reward = get_mape_reward(new_mape, 1)
        mae_reward  = get_mae_reward(q_mae, new_mae, 2)
        combined_reward = mae_reward + rank_reward
        # mae_lst.append(new_mae)
        rewards.append(combined_reward)
    # return rewards, mae_lst
    return rewards

def evaluate_agent(agent, test_states, test_bm_preds, test_X):
    with torch.no_grad():
        weights = agent.select_action(test_states)  # (2816, 9)
    act_counter = Counter(weights.argmax(1))
    act_sorted  = sorted([(k, v) for k, v in act_counter.items()])
    weights = np.expand_dims(weights, -1)  # (2816, 9, 1)
    weighted_y = weights * test_bm_preds  # (2816, 9, 24)
    weighted_y = weighted_y.sum(1)  # (2816, 24)
    mae_loss = mean_absolute_error(test_X, weighted_y)
    mape_loss = mean_absolute_percentage_error(test_X, weighted_y)
    return mae_loss, mape_loss, act_sorted

def evaluate_agent_test(agent, train_states, train_bm_preds, test_states, test_bm_preds, test_y, anomaly_ratio):
    with torch.no_grad():
        weights_train = agent.select_action(train_states)  # (2816, 9)
        weights_test = agent.select_action(test_states)  # (2816, 9)

    weights_train = np.expand_dims(weights_train, -1)  # (2816, 9, 1)
    weights_test = np.expand_dims(weights_test, -1)  # (2816, 9, 1)

    weighted_train_y = weights_train * train_bm_preds  # (2816, 9, 24)
    weighted_test_y = weights_test * test_bm_preds  # (2816, 9, 24)

    weighted_train_y = weighted_train_y.sum(1)  # (2816, 24)
    weighted_test_y = weighted_test_y.sum(1)  # (2816, 24)

    # Accuracy Precision Recall Fscore
    pred = (weighted_test_y > threshold).astype(int)
    gt = test_y.astype(int)
    combined_energy = np.concatenate([weighted_train_y, weighted_test_y], axis=0)
    threshold = np.percentile(combined_energy, 100 - anomaly_ratio)

    gt, pred = adjustment(gt, pred) #gt == label
    gt, pred = np.array(pred), np.array(gt)

    accuracy = accuracy_score(gt, pred)
    precision, recall, f_score, support = precision_recall_fscore_support(gt, pred, average='binary')
    return accuracy, precision, recall, f_score

def sparse_explore(obs, act_dim):
    N = len(obs)
    x = np.zeros((N, act_dim))
    randn_idx = np.random.randint(0, act_dim, size=(N,))
    x[np.arange(N), randn_idx] = 1

    # disturb from the vertex
    delta = np.random.uniform(0.02, 0.1, size=(N, 1))
    x[np.arange(N), randn_idx] -= delta.squeeze()

    # noise
    noise = np.abs(np.random.randn(N, act_dim))
    noise[np.arange(N), randn_idx] = 0
    noise /= noise.sum(1, keepdims=True)
    noise = delta * noise
    sparse_action = x + noise

    return sparse_action

def get_state_weight(train_error):
    L = len(train_error)
    best_model = train_error.argmin(1)
    best_model_counter = Counter(best_model)
    model_weight = {k:v/L for k,v in best_model_counter.items()}
    return model_weight