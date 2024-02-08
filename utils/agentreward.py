import torch
import numpy as np
from collections import Counter
from sktime.performance_metrics.forecasting import \
    mean_absolute_error, mean_absolute_percentage_error

def inv_trans(data,std,mean):
     return data * std + mean

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
def get_rank_reward(rank, R=1):
        reward = -R + 2*R*(9 - rank)/9
        return reward

def get_batch_rewards(env, idxes, actions):
    rewards = []
    mae_lst = []
    for i in range(len(idxes)):
        rank, new_mape, new_mae = env.reward_func(idxes[i], actions[i])
        rank_reward = get_rank_reward(rank, 1)
        mape_reward = get_mape_reward(new_mape, 1)
        # mae_reward  = get_mae_reward(q_mae, new_mae, 2)
        combined_reward = mape_reward + rank_reward
        mae_lst.append(new_mae)
        rewards.append(combined_reward)
    return rewards, mae_lst

def evaluate_agent(agent, test_states, test_ranks, test_bm_preds, test_y, std, mean):
    with torch.no_grad():
        weights = agent.select_action(test_states, test_ranks)  # (2816, 9)
    act_counter = Counter(weights.argmax(1))
    act_sorted  = sorted([(k, v) for k, v in act_counter.items()])
    weights = np.expand_dims(weights, -1)  # (2816, 9, 1)
    weighted_y = weights * test_bm_preds  # (2816, 9, 24)
    weighted_y = weighted_y.sum(1)  # (2816, 24)
    mae_loss = mean_absolute_error(test_y), inv_trans(weighted_y, std, mean)
    mape_loss = mean_absolute_percentage_error(inv_trans(test_y,  std, mean), inv_trans(weighted_y,  std, mean))
    return mae_loss, mape_loss, act_sorted