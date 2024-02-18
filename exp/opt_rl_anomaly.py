# from data_provider.data_factory import data_provider
# from utils.metrics import NegativeCorr
# from utils.agentreward import sparse_explore, get_batch_rewards, get_state_weight
# from sktime.performance_metrics.forecasting import mean_absolute_error, mean_absolute_percentage_error
# from models import DDPG
# import numpy as np
# import torch
# import torch.nn as nn
# from torch import optim
# import torch.nn.functional as F
# from scipy.special import softmax
# import tqdm

# import os
# import time

# import warnings
# import matplotlib.pyplot as plt
# import numpy as np

# warnings.filterwarnings('ignore')
# def inv_trans(x):
#     means = x.mean(1, keepdim=True).detach()  # B x 1 x E
#     x = x - means
#     std_enc = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()  # B x 1 x E
#     x = x / std_enc
#     return x

# class ReplayBuffer:
#     def __init__(self, action_dim, device, max_size=int(1e5)):
#         self.max_size = max_size
#         self.device = device
#         self.ptr = 0
#         self.size = 0

#         # In TS data, `next_state` is just the S[i+1]
#         self.states = np.zeros((max_size, 1), dtype=np.int32)
#         self.actions = np.zeros((max_size, action_dim), dtype=np.float32)
#         self.rewards = np.zeros((max_size, 1), dtype=np.float32)

#     def add(self, state, action, reward):
#         self.states[self.ptr] = state
#         self.actions[self.ptr] = action
#         self.rewards[self.ptr] = reward
#         self.ptr = (self.ptr + 1) % self.max_size
#         self.size = min(self.size + 1, self.max_size)

#     def sample(self, batch_size=256):
#         ind = np.random.randint(self.size, size=batch_size)
#         states = self.states[ind].squeeze()
#         actions = torch.FloatTensor(self.actions[ind]).to(self.device)
#         rewards = torch.FloatTensor(self.rewards[ind]).to(self.device)
#         return (states, actions, rewards.squeeze())
# class Env:
#     def __init__(self, train_error, train_y, PATH):
#         self.error = train_error
#         self.bm_preds = np.load(f'{PATH}/bm_train_preds.npy')
#         self.y = train_y
    
#     def reward_func(self, idx, action):
#         if isinstance(action, int):
#             tmp = np.zeros(self.bm_preds.shape[1])
#             tmp[action] = 1.
#             action = tmp
#         weighted_y = np.multiply(action.reshape(-1, 1), self.bm_preds[idx])
#         weighted_y = weighted_y.sum(axis=0)
#         new_mape = mean_absolute_percentage_error(inv_trans(self.y[idx]), inv_trans(weighted_y))
#         new_mae = mean_absolute_error(inv_trans(self.y[idx]), inv_trans(weighted_y))
#         new_error = np.array([*self.error[idx], new_mape])
#         rank = np.where(np.argsort(new_error) == len(new_error) - 1)[0][0]
#         return rank, new_mape, new_mae 
    
# class DDPGAgent:
#     def __init__(self, use_td, states, obs_dim, act_dim, device, hidden_dim=256,
#                  lr=3e-4, gamma=0.99, tau=0.005):
#         # initialize the actor & target_actor
#         self.actor = DDPG.Actor(obs_dim, act_dim, hidden_dim).to(device)
#         self.target_actor = DDPG.Actor(obs_dim, act_dim, hidden_dim).to(device)
#         self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)

#         # initialize the critic
#         self.critic = DDPG.Critic(obs_dim, act_dim, hidden_dim).to(device)
#         self.target_critic = DDPG.Critic(obs_dim, act_dim, hidden_dim).to(device)
#         self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)

#         # training states
#         self.states = states

#         # parameters
#         self.gamma  = gamma
#         self.tau    = tau
#         self.use_td = use_td

#         # update the target network
#         for param, target_param in zip(
#                 self.critic.parameters(), self.target_critic.parameters()):
#             target_param.data.copy_(param.data)
#         for param, target_param in zip(
#                 self.actor.parameters(), self.target_actor.parameters()):
#             target_param.data.copy_(param.data)

#     def select_action(self, obs):
#         with torch.no_grad():
#             action = self.actor(obs).cpu().numpy()
#         return F.softmax(action, axis=1)

#     def update(self,
#                sampled_obs_idxes,
#                sampled_actions,
#                sampled_rewards,
#                sampled_weights=None):
#         batch_obs = self.states[sampled_obs_idxes]  # (512, 7, 20)

#         with torch.no_grad():
#             if self.use_td:
#                 # update w.r.t the TD target
#                 batch_next_obs = self.states[sampled_obs_idxes + 1]
#                 target_q = self.target_critic(
#                     batch_next_obs, self.target_actor(batch_next_obs))  # (B,)
#                 target_q = sampled_rewards + self.gamma * target_q  # (B,)
#             else:
#                 # without TD learning, just is supervised learning
#                 target_q = sampled_rewards
#         current_q = self.critic(batch_obs, sampled_actions)     # (B,)

#         # critic loss
#         if sampled_weights is None:
#             q_loss = F.mse_loss(current_q, target_q)
#         else:
#             # weighted mse loss
#             q_loss = (sampled_weights * (current_q - target_q)**2).sum() /\
#                 sampled_weights.sum()

#         self.critic_optimizer.zero_grad()
#         q_loss.backward()
#         self.critic_optimizer.step()

#         # actor loss ==> convert actor output to softmax weights
#         if sampled_weights is None:
#             actor_loss = -self.critic(
#                 batch_obs, softmax(self.actor(batch_obs), dim=1)).mean()
#         else:
#             # weighted actor loss
#             actor_loss = -self.critic(batch_obs, F.softmax(self.actor(batch_obs), dim=1))
#             actor_loss = (sampled_weights * actor_loss).sum() / sampled_weights.sum()
#         self.actor_optimizer.zero_grad()
#         actor_loss.backward()
#         self.actor_optimizer.step()

#         # Update the frozen target models
#         if self.use_td:
#             for param, target_param in zip(
#                     self.critic.parameters(), self.target_critic.parameters()):
#                 target_param.data.copy_(
#                     self.tau * param.data + (1 - self.tau) * target_param.data)
#         for param, target_param in zip(
#                 self.actor.parameters(), self.target_actor.parameters()):
#             target_param.data.copy_(
#                 self.tau * param.data + (1 - self.tau) * target_param.data)
        
#         return {
#             'q_loss': q_loss.item(),
#             'pi_loss': actor_loss.item(),
#             'current_q': current_q.mean().item(),
#             'target_q': target_q.mean().item()
#         }

# class OPT_RL_Mantra:
#     def __init__(self,args):
#         self.args = args
#         self.device = self._acquire_device()
#         self.agent = self._build_model()
#         self.replayBuffer = None
#         self.extra_buffer = None

#     def _acquire_device(self):
#         if self.args.use_gpu:
#             os.environ["CUDA_VISIBLE_DEVICES"] = str(
#                 self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
#             device = torch.device('cuda:{}'.format(self.args.gpu))
#             print('Use GPU: cuda:{}'.format(self.args.gpu))
#         else:
#             device = torch.device('cpu')
#             print('Use CPU')
#         return device
    
#     def _build_model(self):
#         agent = DDPGAgent(self.args, self.device, self.model_dict)
#         return agent
    
#     def _get_data(self,flag):
#         data_set, data_loader = data_provider(self.args, flag)
#         return data_set, data_loader
    
#     def _initiate_buffer(self, act_dim):
#         self.replayBuffer = ReplayBuffer(act_dim, max_size=int(1e5))
#         self.extra_buffer = ReplayBuffer(act_dim, max_size=int(1e5))
    
#     def _select_optimizer_actor(self):
#         actor_optim = optim.Adam(self.model.actor.parameters(), lr=self.args.learning_rate)
#         return actor_optim
    
#     def _select_optimizer_critic(self):
#         critic_optim = optim.Adam(self.model.critic.parameters(), lr=self.args.learning_rate)
#         return critic_optim

#     def _select_criterion(self):
#         if self.args.loss_type == "negative_corr":
#             criterion = NegativeCorr(self.args.correlation_penalty)
#         else:
#             criterion = nn.MSELoss()
#         return criterion
#     def pretrain_actor(self, obs_dim, act_dim, hidden_dim, states, train_error, cls_weights, 
#                    valid_states, valid_error):
#         best_train_model = torch.LongTensor(train_error.argmin(1)).to(self.device)
#         best_valid_model = torch.LongTensor(valid_error.argmin(1)).to(self.device)

#         actor = DDPG.Actor(self.args, act_dim, obs_dim).to(self.device)
#         best_actor = DDPG.Actor(self.args, act_dim, obs_dim).to(self.device)
#         cls_weights = torch.FloatTensor([1/cls_weights[w] for w in range(act_dim)]).to(self.device)

#         L = len(states)
#         batch_size = 512
#         batch_num  = int(np.ceil(L / batch_size))
#         optimizer  = torch.optim.Adam(actor.parameters(), lr=3e-4)
#         loss_fn    = nn.CrossEntropyLoss(weight=cls_weights)  # weighted CE loss
#         best_acc   = 0
#         patience   = 0
#         max_patience = 5
#         for epoch in tqdm.trange(200, desc='[Pretrain]'):
#             epoch_loss = []
#             shuffle_idx = np.random.permutation(np.arange(L))
#             for i in range(batch_num):
#                 batch_idx = shuffle_idx[i*batch_size: (i+1)*batch_size]
#                 optimizer.zero_grad()
#                 batch_out = actor(states[batch_idx])
#                 loss = loss_fn(batch_out, best_train_model[batch_idx])
#                 loss.backward()
#                 optimizer.step()
#                 epoch_loss.append(loss.item())
#             with torch.no_grad():
#                 pred = actor(valid_states)
#                 pred_idx = pred.argmax(1)
#                 acc = (pred_idx == best_valid_model).sum() / len(pred)
#             print(f'# epoch {epoch+1}: loss = {np.average(epoch_loss):.5f}\tacc = {acc:.3f}')

#             # early stop w.r.t. validation acc
#             if acc > best_acc:
#                 best_acc = acc
#                 patience = 0
#                 # update best model
#                 for param, target_param in zip(
#                         actor.parameters(), best_actor.parameters()):
#                     target_param.data.copy_(param.data)
#             else:
#                 patience += 1
            
#             if patience == max_patience:
#                 break

#         with torch.no_grad():
#             pred = best_actor(valid_states)
#             pred_idx = pred.argmax(1)
#             acc = (pred_idx == best_valid_model).sum() / len(pred)    
#         print(f'valid acc for pretrained actor: {acc:.3f}') 
#         return best_actor
 
#     def active_urt_reinforcment_learning(self, setting):
#         train_data, train_loader = self._get_data(flag='train')
#         vali_data, vali_loader = self._get_data(flag='val')
#         test_data, test_loader = self._get_data(flag='test')
        
#         # train_error, valid_error = 
#         # train_preds, valid_preds, test_preds = 

#         states, train_y = next(iter(train_loader)) #train_X
#         valid_states = next(iter(vali_loader))[0]
#         test_states = next(iter(test_loader))[0]

#         train_X = next(iter(train_data))[0]
#         L = len(train_X) - 1 if self.args.use_td else len(train_X)

#         obs_dim = states.shape[2] # Mengambil Feature
#         act_dim = train_error.shape[-1]
#         self._initiate_buffer(act_dim)

#         env = Env(train_error, train_y) #train_error, train_y, bm_train_preds itu punya features 9
#         best_model_weight = get_state_weight(train_error)
#         state_weights = [1/best_model_weight[i] for i in train_error.argmin(1)]
#         if self.args.use_weight:
#             state_weights = torch.FloatTensor(state_weights).to(device)
#         else:
#             state_weights = None
#         if self.args.use_pretrain:
#             pretrained_actor = self.pretrain_actor(obs_dim,
#                                             act_dim,
#                                             hidden_dim=100,
#                                             states=states,
#                                             train_error=train_error, 
#                                             cls_weights=best_model_weight,
#                                             valid_states=valid_states, 
#                                             valid_error=valid_error
#                                             )
            
#             # copy the pretrianed actor 
#             for param, target_param in zip(
#                     pretrained_actor.parameters(), agent.actor.parameters()):
#                 target_param.data.copy_(param.data)
#             for param, target_param in zip(
#                     pretrained_actor.parameters(), agent.target_actor.parameters()):
#                 target_param.data.copy_(param.data)
#         step_num  = int(np.ceil(L / self.args.step_size))
#         for epoch in tqdm.trange(self.args.train_epochs):
#             q_loss_lst, pi_loss_lst, q_lst, target_q_lst  = [], [], [], []
#             t1 = time.time()
#             shuffle_idx = np.random.permutation(np.arange(L))
#             for i in range(step_num):
#                 batch_idx = shuffle_idx[i*self.args.step_size: (i+1)*self.args.step_size]        # (512,)
#                 batch_states = states[batch_idx]
#                 if np.random.random() < epsilon:
#                     batch_actions = sparse_explore(batch_states, act_dim)
#                 else:
#                     batch_actions = self.agent.select_action(batch_states)
#                 batch_rewards, batch_mae = get_batch_rewards(env, batch_idx, batch_actions)
#                 for j in range(len(batch_idx)):
#                     self.replay_buffer.add(batch_idx[j], batch_actions[j], batch_rewards[j])
#                     if self.args.use_extra and batch_rewards[j] <= -1.:
#                         self.extra_buffer.add(batch_idx[j], batch_actions[j], batch_rewards[j])

#                 sampled_obs_idxes, sampled_actions, sampled_rewards = self.replay_buffer.sample(512)
#                 if self.args.use_weight:
#                     sampled_weights = state_weights[sampled_obs_idxes]
#                 else:
#                     sampled_weights = None
#                 info = self.agent.forward(sampled_obs_idxes, sampled_actions, sampled_rewards, sampled_weights)

#                 pi_loss_lst.append(info['pi_loss'])
#                 q_loss_lst.append(info['q_loss'])
#                 q_lst.append(info['current_q'])
#                 target_q_lst.append(info['target_q'])

#                 if use_extra and extra_buffer.ptr > 512:
#                     sampled_obs_idxes, sampled_actions, sampled_rewards = extra_buffer.sample(512)
#                     if use_weight:
#                         sampled_weights = state_weights[sampled_obs_idxes]
#                     else:
#                         sampled_weights = None
#                     info = agent.update(sampled_obs_idxes, sampled_actions, sampled_rewards, sampled_weights)
#                     pi_loss_lst.append(info['pi_loss'])
#                     q_loss_lst.append(info['q_loss'])
#                     q_lst.append(info['current_q'])
#                     target_q_lst.append(info['target_q'])

#             valid_mae_loss, valid_mape_loss, count_lst = evaluate_agent(agent, valid_states, valid_preds, valid_y) #ERROR
#             print(f'\n# Epoch {epoch + 1} ({(time.time() - t1)/60:.2f} min): '
#                 f'valid_mae_loss: {valid_mae_loss:.3f}\t'
#                 f'valid_mape_loss: {valid_mape_loss*100:.3f}\t' 
#                 f'q_loss: {np.average(q_loss_lst):.5f}\t'
#                 f'current_q: {np.average(q_lst):.5f}\t'
#                 f'target_q: {np.average(target_q_lst):.5f}\n')

#             if valid_mape_loss < best_mape_loss:
#                 best_mape_loss = valid_mape_loss
#                 patience = 0
#                 # save best model
#                 for param, target_param in zip(agent.actor.parameters(), best_actor.parameters()):
#                     target_param.data.copy_(param.data)
#             else:
#                 patience += 1
#             if patience == max_patience:
#                 break
#             epsilon = max(epsilon-0.2, 0.1)