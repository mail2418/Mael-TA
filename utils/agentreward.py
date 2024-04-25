import torch
import numpy as np
from collections import Counter
from sktime.performance_metrics.forecasting import \
    mean_absolute_error, mean_absolute_percentage_error
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from utils.tools import adjustment
import math
from gym.utils import seeding
from gym import spaces
import gym
import random
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
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

    list_weighted_y = []
    for i in range(math.ceil(test_bm_preds.shape[0]/weights.shape[0])):
        start_idx = i * weights.shape[0]
        end_idx = (i + 1) * weights.shape[0]
        chunk = test_bm_preds[start_idx:end_idx]
        weighted_sum = np.multiply(weights, chunk).sum(1)
        list_weighted_y.append(weighted_sum)
    weighted_y = np.concatenate(list_weighted_y, axis=0)
    # weighted_y = weights * test_bm_preds[:weights.shape[0]]  # (2816, 9, 24)
    # weighted_y = weighted_y.sum(1)  # (2816, 24)
    mae_loss = mean_absolute_error(test_X, weighted_y)
    mape_loss = mean_absolute_percentage_error(test_X, weighted_y)
    return mae_loss, mape_loss, act_sorted

def evaluate_agent_test(agent, train_states, train_bm_preds, test_states, test_bm_preds, test_labels, anomaly_ratio):
    with torch.no_grad():
        weights_train = agent.select_action(train_states)  # (58120, 3) #outputs train pada test misal = 32,100,55
        weights_test = agent.select_action(test_states)  # (58120, 3)

    weights_train = np.expand_dims(weights_train, -1)  # (58120, 3, 1)
    weights_test = np.expand_dims(weights_test, -1)  # (58120, 3, 1)

    weighted_train_y = weights_train * train_bm_preds  # (58120, 3, 55)
    weighted_test_y = weights_test * test_bm_preds  # (58120, 3, 55)

    weighted_train_y = weighted_train_y.sum(1) # (58120)
    weighted_test_y = weighted_test_y.sum(1)# (58120)
    # weighted_test_y = test_bm_preds.mean(2).sum(1)# (58120)

    # Accuracy Precision Recall Fscore
    combined_energy = np.concatenate([weighted_train_y, weighted_test_y], axis=0).reshape(-1)

    threshold = np.percentile(combined_energy, 100 - anomaly_ratio)

    print("Threshold :", threshold)

    gt = test_labels[:,:weighted_test_y.shape[1]].reshape(-1).astype(int)
    weighted_test_y = weighted_test_y.reshape(-1)
    # weighted_test_y = weighted_test_y.reshape(-1)
    pred = (weighted_test_y > threshold).astype(int)
    # pred = (test_bm_preds > threshold).astype(int)

    gt, pred = adjustment(gt, pred) #gt == label
    gt, pred = np.array(gt), np.array(pred)
    print("pred: ", pred.shape)
    print("gt:   ", gt.shape)

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

class EnvOffline_dist_conf(gym.Env):
    '''Gym environment for model selection in offline setting.

    model_path: path to the pretrained models;
    list_pred_sc: the flattened list of raw predicted scores (each one being 1D numpy array) 
                    of the testing data from each model;
    list_thresholds: the list of raw anomaly thresholds from each model;'''
    
    def __init__(self, list_pred_sc, list_thresholds,list_gtruth, model_path="./base_detectors"):

        # Length of the testing data, number of models
        self.len_data = len(list_pred_sc[0])
        self.num_models = len(list_pred_sc)

        #List of ground truth labels
        self.gtruth = list_gtruth
           
        # Raw scores and thresholds of the testing data
        self.list_pred_sc = list_pred_sc
        self.list_thresholds = list_thresholds

        # Scale the raw scores/thresholds and save each scaler
        self.scaler = []
        self.list_scaled_sc = []
        self.list_scaled_thresholds = []
        for i in range(self.num_models):
            scaler_tmp = MinMaxScaler()
            self.list_scaled_sc.append(scaler_tmp.fit_transform(self.list_pred_sc[i].reshape(-1,1)))
            self.scaler.append(scaler_tmp)
            self.list_scaled_thresholds.append(scaler_tmp.transform(self.list_thresholds[i].reshape(-1,1)))

        # Extract predictions
        self.list_pred = []
        for i in range(self.num_models):
            pred_tmp = np.zeros(self.len_data)
            for length in range(self.len_data):
                if self.list_scaled_sc[i][length] > self.list_scaled_thresholds[i]:
                    pred_tmp[length] = 1
            self.list_pred.append(pred_tmp)

        # Extract distance-to-threshold confidence
        self.dist_conf=[]
        for length in range(self.len_data):
            dist_tmp = []
            for i in range(self.num_models):
                dist_tmp.append(self.list_scaled_sc[i][length] - self.list_scaled_thresholds[i])
            self.dist_conf.append(dist_tmp)
        

        # Gym settings
        self.action_space = spaces.Discrete(self.num_models) 
        # state_dim is 4 , each corresponds to scaled_sc, scaled_thresholds, pred, dist_conf 
        self.observation_space = spaces.Box(low=0, high=1, shape=(4, ), dtype=np.float32)
        self.seed()
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def render(self):
        pass

class TrainEnvOffline_dist_conf(EnvOffline_dist_conf):
    '''The training environment in offline setting.

        list_gtruth: the list of ground truth labels (each one being 1D numpy array) of the 
                    testing data by each models.'''

    def __init__(self, list_pred_sc, list_thresholds, list_gtruth):
        super().__init__(list_pred_sc, list_thresholds, list_gtruth)
    
    def reset(self):
        self.pointer = 0 # Reset the pointer to the beginning of the testing data
        self.done = False
        return self._get_state()

    def step(self, action):
        '''Return:
            observation: the current state of the environment;
            reward: the reward of the action;
            done: whether the episode is over;'''

        # Get the current state
        observation = self._get_state(action)

        # Get the reward
        reward=self._get_reward(observation)

        self.pointer += 1

        # Check whether the episode is over
        if self.pointer >= self.len_data:
            self.done = True
        else:
            self.done = False

        return observation, reward, self.done, {}

    def _get_state(self,action=None):

        '''Return:
            observation: the current state of the environment.'''

        if self.pointer==0: # If the pointer is at the beginning of the testing data
            action=random.randint(0,self.num_models-1) # Randomly select a model

        # Get the current state
        observation = np.zeros(4) # 4 dims - scaled scores, scaled thresholds, labels, dist_conf
        observation[0] = self.list_scaled_sc[action][self.pointer]
        observation[1] = self.list_scaled_thresholds[action]
        observation[2] = self.list_pred[action][self.pointer]
        observation[3] = self.dist_conf[self.pointer][action]

        return observation

    def _get_reward(self,observation):
        '''Return:
            reward: the reward of the action.'''

        # Get the reward
        if self.gtruth[self.pointer]==1: # If the ground truth is 1 anomaly
            if observation[2]==1: # If the model predicts 1 anomaly correctly - True Positive (TP)
                reward = 1
            else: # If the model predicts 0 normal incorrectly - False Negative (FN)
                reward = -1.5
        else: # If the ground truth is 0 normal
            if observation[2]==1: # If the model predicts 1 anomaly incorrectly - False Positive (FP)
                reward = -0.4
            else: # If the model predicts 0 normal correctly - True Negative (TN)
                reward = 0.1

        return reward

def eval_model(model,env):
    '''Evaluate the model on the environment.

        model: the model to be evaluated;
        env: the environment to be evaluated on.
        
        Return:
            precision: the precision of the model;
            recall: the recall of the model;
            f1: the f1 score of the model;
            conf_matrix: the confusion matrix of the model, comparing it with the ground truth;
            preds: the list of predictions of the model.'''

    # The ground truth labels
    gtruth = env.gtruth

    # Reset the environment
    observation = env.reset()

    # Evaluate the model - get predicted labels and total reward
    preds = []
    while True:
        action = model.predict(observation)
        observation, reward, done, _ = env.step(action[0]) # action[0] is the index of the action, action is a tuple
        preds.append(observation[2])
        if done:
            break
    
    prec=precision_score(gtruth,preds,pos_label=1)
    rec=recall_score(gtruth,preds,pos_label=1)
    f1=f1_score(gtruth,preds,pos_label=1)
    conf_matrix=confusion_matrix(gtruth,preds,labels=[0,1])

    return prec,rec,f1,conf_matrix, preds
