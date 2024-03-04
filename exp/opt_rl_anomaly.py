from utils.datapredsrl import load_data_rl
from utils.agentreward import sparse_explore, get_batch_rewards, get_state_weight,evaluate_agent, evaluate_agent_test
from sktime.performance_metrics.forecasting import mean_absolute_error, mean_absolute_percentage_error
from models import DDPG
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.special import softmax
import tqdm
import os
import time
import pandas as pd
import csv
import warnings
import matplotlib.pyplot as plt
import numpy as np


warnings.filterwarnings('ignore')

def scale_input(x):
    means = x.mean(1, keepdim=True).detach()  # B x 1 x E
    x = x - means
    std_enc = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()  # B x 1 x E
    x = x / std_enc
    return x

class ReplayBuffer:
    def __init__(self, action_dim, device, max_size=int(1e5)):
        self.max_size = max_size
        self.device = device
        self.ptr = 0
        self.size = 0

        # In TS data, `next_state` is just the S[i+1]
        self.states = np.zeros((max_size, 1), dtype=np.int8)
        self.actions = np.zeros((max_size, action_dim), dtype=np.float16)
        self.rewards = np.zeros((max_size, 1), dtype=np.float16)

    def add(self, state, action, reward):
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size=256):
        ind = np.random.randint(self.size, size=batch_size)
        states = self.states[ind].squeeze()
        actions = torch.FloatTensor(self.actions[ind]).to(self.device)
        rewards = torch.FloatTensor(self.rewards[ind]).to(self.device)
        return (states, actions, rewards.squeeze())
class Env:
    def __init__(self, train_preds, train_error, train_y):
        self.error = train_error
        self.bm_preds = train_preds
        self.y = train_y
    
    def reward_func(self, idx, action):
        if isinstance(action, int):
            tmp = np.zeros(self.bm_preds.shape[1])
            tmp[action] = 1.
            action = tmp
        weighted_y = np.multiply(action.reshape(-1, 1), self.bm_preds[idx])
        weighted_y = weighted_y.sum(axis=0)
        # new_mape = mean_absolute_percentage_error(inv_trans(self.y[idx]), inv_trans(weighted_y))
        new_mae = mean_absolute_error(self.y[idx], weighted_y)
        new_error = np.array([*self.error[idx], new_mae])
        rank = np.where(np.argsort(new_error) == len(new_error) - 1)[0][0]
        # return rank, new_mape, new_mae 
        return rank, new_mae
    
class DDPGAgent:
    def __init__(self, args, device, lr=3e-4, gamma=0.99, tau=0.005):
        self.args = args
        self.device = device
        self.obs_dim = None
        self.act_dim = None
        self.states = None
        self.lr = lr
        self.gamma  = gamma
        self.tau    = tau
        self.use_td = args.use_td
    def _set_obs_dim(self, obs_dim):
        self.obs_dim = obs_dim
    def _set_act_dim(self, act_dim):
        self.act_dim = act_dim
    def _set_states(self, states):
        self.states = states
    def _init_actor(self):
        self.actor = DDPG.Actor(self.args, self.act_dim, self.obs_dim).to(self.device)
        self.target_actor = DDPG.Actor(self.args, self.act_dim, self.obs_dim).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
    def _init_critic(self):
        self.critic = DDPG.Critic(self.args, self.act_dim, self.obs_dim).to(self.device)
        self.target_critic = DDPG.Critic(self.args, self.act_dim, self.obs_dim).to(self.device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.lr)
    def _update_network(self):
        for param, target_param in zip(
                self.critic.parameters(), self.target_critic.parameters()):
            target_param.data.copy_(param.data)
        for param, target_param in zip(
                self.actor.parameters(), self.target_actor.parameters()):
            target_param.data.copy_(param.data)

    def select_action(self, obs):
        with torch.no_grad():
            action = self.actor(obs).cpu().numpy()
        return softmax(action, axis=1)

    def update(self,sampled_obs_idxes,sampled_actions,sampled_rewards,sampled_weights=None):
        batch_obs = self.states[sampled_obs_idxes]  # (512, 7, 20)
        with torch.no_grad():
            if self.use_td:
                # update w.r.t the TD target
                batch_next_obs = self.states[sampled_obs_idxes + 1]
                target_actor_output = self.target_actor(batch_next_obs)

                target_critic_output = self.target_critic(batch_next_obs, target_actor_output)  # (B,)
                target_critic_output = sampled_rewards + self.gamma * target_critic_output  # (B,)
            else:
                # without TD learning, just is supervised learning
                target_critic_output = sampled_rewards
            
        current_q = self.critic(batch_obs, sampled_actions)     # (B,)
        # critic loss
        if sampled_weights is None:
            q_loss = F.mse_loss(current_q, target_critic_output)
        else:
            # weighted mse loss
            q_loss = (sampled_weights * (current_q - target_critic_output)**2).sum() /\
                sampled_weights.sum()

        self.critic_optimizer.zero_grad()
        q_loss.backward()
        self.critic_optimizer.step()

        # actor loss ==> convert actor output to softmax weights
        if sampled_weights is None:
            actor_loss = -self.critic(
                batch_obs, F.softmax(self.actor(batch_obs), dim=1)).mean()
        else:
            # weighted actor loss
            actor_loss = -self.critic(batch_obs, F.softmax(self.actor(batch_obs), dim=1))
            actor_loss = (sampled_weights * actor_loss).sum() / sampled_weights.sum()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update the frozen target models
        if self.use_td:
            for param, target_param in zip(
                    self.critic.parameters(), self.target_critic.parameters()):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data)
        for param, target_param in zip(
                self.actor.parameters(), self.target_actor.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data)
        
        return {
            'q_loss': q_loss.item(),
            'pi_loss': actor_loss.item(),
            'current_q': current_q.mean().item(),
            'target_q': target_critic_output.mean().item()
        }

class OPT_RL_Mantra:
    def __init__(self,args):
        self.args = args
        self.device = self._acquire_device()
        self.agent = self._build_model()
        self.replay_buffer = None
        self.extra_buffer = None

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device
    
    def _build_model(self):
        agent = DDPGAgent(self.args, self.device)
        return agent
    
    def _initiate_buffer(self, act_dim):
        self.replay_buffer = ReplayBuffer(act_dim, self.device, max_size=int(1e5))
        self.extra_buffer = ReplayBuffer(act_dim, self.device, max_size=int(1e5))
    
    def pretrain_actor(self, obs_dim, act_dim, states, train_error, cls_weights, 
                   valid_states, valid_error):
        best_train_model = torch.LongTensor(train_error.argmin(1)).to(self.device)
        best_valid_model = torch.LongTensor(valid_error.argmin(1)).to(self.device)

        memory_allocated = torch.cuda.memory_allocated()
        print(f"Memory Allocated: {memory_allocated / 1024**3:.2f} GB")

        # Get the peak amount of memory allocated on the GPU
        max_memory_allocated = torch.cuda.max_memory_allocated()
        print(f"Peak Memory Allocated: {max_memory_allocated / 1024**3:.2f} GB")

        actor = DDPG.Actor(self.args, act_dim, obs_dim).to(self.device)
        best_actor = DDPG.Actor(self.args, act_dim, obs_dim).to(self.device)
        cls_weights = torch.FloatTensor([1/cls_weights[w] for w in range(act_dim)]).to(self.device)

        L = len(states)
        batch_size = 512
        batch_num  = int(np.ceil(L / batch_size))
        optimizer  = torch.optim.Adam(actor.parameters(), lr=3e-4)
        loss_fn    = nn.CrossEntropyLoss(weight=cls_weights)  # weighted CE loss
        best_acc   = 0
        patience   = 0
        max_patience = 5
        for epoch in tqdm.trange(200, desc='[Pretrain]'):
            epoch_loss = []
            shuffle_idx = np.random.permutation(np.arange(L))
            for i in range(batch_num):
                batch_idx = shuffle_idx[i*batch_size: (i+1)*batch_size]
                optimizer.zero_grad()
                batch_out = actor(states[batch_idx])
                loss = loss_fn(batch_out, best_train_model[batch_idx])
                loss.backward()
                optimizer.step()
                epoch_loss.append(loss.item())
            with torch.no_grad():
                pred = actor(valid_states)
                pred_idx = pred.argmax(1)
                acc = (pred_idx == best_valid_model).sum() / len(pred)
            print(f'# epoch {epoch+1}: loss = {np.average(epoch_loss):.5f}\tacc = {acc:.3f}')

            # early stop w.r.t. validation acc
            if acc > best_acc:
                best_acc = acc
                patience = 0
                # update best model
                for param, target_param in zip(actor.parameters(), best_actor.parameters()):
                    target_param.data.copy_(param.data)
            else:
                patience += 1
            
            if patience == max_patience:
                break

        with torch.no_grad():
            pred = best_actor(valid_states)
            pred_idx = pred.argmax(1)
            acc = (pred_idx == best_valid_model).sum() / len(pred)    
        print(f'valid acc for pretrained actor: {acc:.3f}') 
        return best_actor
 
    def active_urt_reinforcment_learning(self, setting):
        epsilon = self.args.epsilon
        path_ds = os.path.join(self.args.root_path, setting)
        train_X, valid_X, test_X, train_y, valid_y, test_labels, train_error, valid_error, _ = load_data_rl(self.args.root_path, setting)
        train_preds, valid_preds, test_preds = np.load(f'{path_ds}/bm_train_preds_new.npy', allow_pickle=True),np.load(f'{path_ds}/bm_valid_preds_new.npy', allow_pickle=True),np.load(f'{path_ds}/bm_test_preds_new.npy', allow_pickle=True)

        train_X = np.swapaxes(train_X, 2, 1).astype(np.float16)
        valid_X = np.swapaxes(valid_X, 2, 1).astype(np.float16)
        test_X  = np.swapaxes(test_X,  2, 1).astype(np.float16)
        train_y = train_y.astype(np.float16)
        valid_y = valid_y.astype(np.float16)

        train_error = train_error.astype(np.float16)
        valid_error = valid_error.astype(np.float16)

        train_preds = train_preds.astype(np.float16)
        valid_preds = valid_preds.astype(np.float16)
        test_preds = test_preds.astype(np.float16)

        L = len(train_X) - 1 if self.args.use_td else len(train_X)
        states = torch.FloatTensor(train_X).to(self.device)
        valid_states = torch.FloatTensor(valid_X).to(self.device)
        test_states = torch.FloatTensor(test_X).to(self.device)

        obs_dim = states.shape[1] # Mengambil Feature
        act_dim = train_error.shape[-1]
        
        env = Env(train_preds,train_error, train_y)
        best_model_weight = get_state_weight(train_error)

        state_weights = [1/best_model_weight[i] for i in train_error.argmin(1)]
        if self.args.use_weight:
            state_weights = torch.FloatTensor(state_weights).to(self.device)
        else:
            state_weights = None
        
        if not os.path.exists(f'{path_ds}/batch_buffer.csv'):
            batch_buffer = []
            for state_idx in tqdm.trange(L, desc='[Create buffer]'):
                best_model_idx = train_error[state_idx].argmin()
                for action_idx in range(act_dim):
                    rank, mae = env.reward_func(state_idx, action_idx)
                    batch_buffer.append((state_idx, action_idx, rank, mae, best_model_weight[best_model_idx]))
            batch_buffer_df = pd.DataFrame(
                batch_buffer,
                columns=['state_idx', 'action_idx', 'rank', 'mae', 'weight']) 
            batch_buffer_df.to_csv(f'{path_ds}/batch_buffer.csv')
        else:
            batch_buffer_df = pd.read_csv(f'{path_ds}/batch_buffer.csv', index_col=0)

        q_mae = [batch_buffer_df['mae'].quantile(0.1*i) for i in range(1, 10)] 
        
        # INIT AGENT AND REPLAY EXTRA BUFFERS
        self._initiate_buffer(act_dim)
        self.agent._set_states(states)
        self.agent._set_obs_dim(states.shape[1])
        self.agent._set_act_dim(act_dim)
        self.agent._init_actor()
        self.agent._init_critic()
        self.agent._update_network()
        
        #Error dari pretrain apabila pakai dataset besar
        # mengcopy parameter dari pretrain_actor ke agent actor
        if self.args.use_pretrain:
            pretrained_actor = self.pretrain_actor(obs_dim,
                                            act_dim,
                                            states=states,
                                            train_error=train_error, 
                                            cls_weights=best_model_weight,
                                            valid_states=valid_states, 
                                            valid_error=valid_error
                                            )
            # copy the pretrianed actor 
            for param, target_param in zip(
                    pretrained_actor.parameters(), self.agent.actor.parameters()):
                target_param.data.copy_(param.data)
            for param, target_param in zip(
                    pretrained_actor.parameters(), self.agent.target_actor.parameters()):
                target_param.data.copy_(param.data)
        
        # to save the best model
        best_actor = DDPG.Actor(self.args, act_dim, obs_dim).to(self.device)
        for param, target_param in zip(self.agent.actor.parameters(), best_actor.parameters()):
            target_param.data.copy_(param.data)
        
        # warm up
        for _ in tqdm.trange(200, desc='[Warm Up]'):
            shuffle_idxes   = np.random.randint(0, L, 300)
            sampled_states  = states[shuffle_idxes] 
            sampled_actions = self.agent.select_action(sampled_states)
            #sampled_rewards ===> mae_reward + rank_reward
            sampled_rewards = get_batch_rewards(env, shuffle_idxes, sampled_actions, q_mae) 
            for i in range(len(sampled_states)):
                self.replay_buffer.add(shuffle_idxes[i], sampled_actions[i], sampled_rewards[i])

                if self.args.use_extra and sampled_rewards[i] <= -1.:
                    self.extra_buffer.add(shuffle_idxes[i], sampled_actions[i], sampled_rewards[i])
        
        best_mae_loss = np.inf
        patience, max_patience = 0, 5
        step_num  = int(np.ceil(L / self.args.step_size))

        #TRAINING
        for epoch in tqdm.trange(self.args.train_epochs_rl):
            q_loss_lst, pi_loss_lst, q_lst, target_q_lst  = [], [], [], []
            t1 = time.time()
            shuffle_idx = np.random.permutation(np.arange(L))
            for i in range(step_num):
                batch_idx = shuffle_idx[i*self.args.step_size : (i+1)*self.args.step_size]        # (512,)
                batch_states = states[batch_idx]
                if np.random.random() < epsilon:
                    # membuat matriks tindakan (action matrix) yang memiliki struktur yang cukup berserakan (sparse) dengan mempertimbangkan observasi (obs) dan dimensi tindakan (act_dim) yang diberikan
                    batch_actions = sparse_explore(batch_states, act_dim)
                else:
                    batch_actions = self.agent.select_action(batch_states)

                batch_rewards = get_batch_rewards(env, batch_idx, batch_actions, q_mae)
                for j in range(len(batch_idx)):
                    self.replay_buffer.add(batch_idx[j], batch_actions[j], batch_rewards[j])
                    if self.args.use_extra and batch_rewards[j] <= -1.:
                        self.extra_buffer.add(batch_idx[j], batch_actions[j], batch_rewards[j])

                sampled_obs_idxes, sampled_actions, sampled_rewards = self.replay_buffer.sample(512)
                if self.args.use_weight:
                    sampled_weights = state_weights[sampled_obs_idxes]
                else:
                    sampled_weights = None
                
                # LOSS BACKWARD
                info = self.agent.update(sampled_obs_idxes, sampled_actions, sampled_rewards, sampled_weights)
                pi_loss_lst.append(info['pi_loss'])
                q_loss_lst.append(info['q_loss'])
                q_lst.append(info['current_q'])
                target_q_lst.append(info['target_q'])

                if self.args.use_extra and self.extra_buffer.ptr > 512:
                    sampled_obs_idxes, sampled_actions, sampled_rewards = self.extra_buffer.sample(512)
                    if self.args.use_weight:
                        sampled_weights = state_weights[sampled_obs_idxes]
                    else:
                        sampled_weights = None
                    # LOSS BACKWARD
                    info = self.agent.update(sampled_obs_idxes, sampled_actions, sampled_rewards, sampled_weights)
                    pi_loss_lst.append(info['pi_loss'])
                    q_loss_lst.append(info['q_loss'])
                    q_lst.append(info['current_q'])
                    target_q_lst.append(info['target_q'])

            # VALIDATION
            valid_mae_loss, _, count_lst = evaluate_agent(self.agent, valid_states, valid_preds, valid_y) #ERROR
            print(f'\n# Epoch {epoch + 1} ({(time.time() - t1)/60:.2f} min): '
                f'valid_mae_loss: {valid_mae_loss:.3f}\t'
                f'q_loss: {np.average(q_loss_lst):.5f}\t'
                f'current_q: {np.average(q_lst):.5f}\t'
                f'target_q: {np.average(target_q_lst):.5f}\n')

            if valid_mae_loss < best_mae_loss:
                best_mae_loss = valid_mae_loss
                patience = 0
                # save best model
                for param, target_param in zip(self.agent.actor.parameters(), best_actor.parameters()):
                    target_param.data.copy_(param.data)
            else:
                patience += 1
            if patience == max_patience:
                break
            epsilon = max(epsilon-0.2, 0.1)

        for param, target_param in zip(self.agent.actor.parameters(), best_actor.parameters()):
            param.data.copy_(target_param)

        # Testing
        accuracy, precision, recall, f_score = evaluate_agent_test(self.agent, states, train_preds, test_states, test_preds, test_labels, self.args.anomaly_ratio)
        print("Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(
            accuracy, precision,
            recall, f_score))
        # CSV
        f_csv = open("training_mantra_anomaly_detection_rl.csv","a")
        csvreader = csv.writer(f_csv)
        datas = [[setting],["Accuracy","Precision","Recall","F-score"],[round(accuracy,4),round(precision,4),round(recall,4),round(f_score,4)]]
        csvreader.writerows(datas)
        
        #Text
        f = open("result_anomaly_detection_mantra_rl.txt", 'a')
        f.write(setting + "  \n")
        f.write("Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(
            accuracy, precision,
            recall, f_score))
        f.write('\n')
        f.write('\n')
        f.close()
    