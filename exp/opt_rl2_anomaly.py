import os
import torch
import torch.nn as nn
import csv
from exp.exp_basic import Exp_Basic
from data_provider.data_factory import data_provider
from utils.tools import my_kl_loss,adjustment
from utils.slowloss import ssl_loss_v2
import numpy as np
from tqdm import trange
from utils.agentreward import TrainEnvOffline_dist_conf, eval_model
from stable_baselines3 import DQN
from models import MaelNet, KBJNet, DCDetector, ns_Transformer, FEDFormer, TimesNet, MaelNetB1, MaelNetS1, ns_TransformerB1, ns_TransformerS1,AutoFormer,MaelNetS2, AnomalyTransformer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

class OPT_RL_Anomaly():
    def __init__(self,args):
        self.model_dict = {
            "MaelNet"    : MaelNet,
            "MaelNetB1":MaelNetB1,
            "MaelNetS1":MaelNetS1,
            "MaelNetS2":MaelNetS2,
            "KBJNet"     : KBJNet,
            "DCDetector" : DCDetector,
            "NSTransformer": ns_Transformer,
            "NSTransformerB1": ns_TransformerB1,
            "NSTransformerS1": ns_TransformerS1,
            "FEDFormer" : FEDFormer,
            "TimesNet": TimesNet,
            "AutoFormer": AutoFormer,
            "AnomalyTransformer": AnomalyTransformer
        }
        self.args = args
        self.device = self._acquire_device()
        self.model = None
        self.anomaly_criterion = nn.MSELoss(reduce=False)
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
    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader
    
    def calculate_train_energy(self,train_loader,temperature,slow_learner=False):
        attens_energy_train = []
        for i, (batch_x, _) in enumerate(train_loader):
            batch_x = batch_x.float().to(self.device)
            if slow_learner:
                s0,s1,s2 = batch_x.shape
                randuniform = torch.empty(s0,s1,s2).uniform_(0, 1)
                m_ones = torch.ones(s0,s1,s2)
                slow_mark = torch.bernoulli(randuniform)
                batch_x_slow = batch_x.clone()
                batch_x_slow = batch_x_slow * (m_ones-slow_mark).to(self.device)

                output, series, prior = self.model(batch_x_slow)
                loss = torch.mean(ssl_loss_v2(output, batch_x_slow, slow_mark, s1, s2, self.device), dim=-1)
            else:
                if self.model.name == "DCDetector":
                    series, prior = self.model(batch_x)
                else:
                    output, series, prior = self.model(batch_x)
                    loss = torch.mean(self.anomaly_criterion(output, batch_x), dim=-1)
            series_loss = 0.0
            prior_loss = 0.0
            for u in range(len(prior)):
                if u == 0:
                    series_loss = my_kl_loss(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.args.win_size)).detach()) * temperature
                    prior_loss = my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.args.win_size)),
                        series[u].detach()) * temperature
                else:
                    series_loss += my_kl_loss(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.args.win_size)).detach()) * temperature
                    prior_loss += my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.args.win_size)),
                        series[u].detach()) * temperature
            metric = torch.softmax((-series_loss - prior_loss), dim=-1)
            if self.model.name == "DCDetector":
                cri = metric.detach().cpu().numpy()
            else: 
                cri = (metric*loss).detach().cpu().numpy()
            attens_energy_train.append(cri)
        attens_energy_train = np.concatenate(attens_energy_train, axis=0).reshape(-1)
        train_energy = np.array(attens_energy_train)
        return train_energy

    def calculate_test_energy(self,test_loader,temperature,slow_learner=False):
        attens_energy_test = []
        test_labels = []
        for i, (batch_x, labels) in enumerate(test_loader):
            test_labels.append(labels)
            batch_x = batch_x.float().to(self.device)
            if slow_learner:
                s0,s1,s2 = batch_x.shape
                randuniform = torch.empty(s0,s1,s2).uniform_(0, 1)
                m_ones = torch.ones(s0,s1,s2)
                slow_mark = torch.bernoulli(randuniform)
                batch_x_slow = batch_x.clone()
                batch_x_slow = batch_x_slow * (m_ones-slow_mark).to(self.device)

                output, series, prior = self.model(batch_x_slow)
                loss = torch.mean(ssl_loss_v2(output, batch_x_slow, slow_mark, s1, s2, self.device), dim=-1)
            else:
                if self.model.name == "DCDetector":
                    series, prior = self.model(batch_x)
                else:
                    output, series, prior = self.model(batch_x)
                    loss = torch.mean(self.anomaly_criterion(output, batch_x), dim=-1)
            series_loss = 0.0
            prior_loss = 0.0
            for u in range(len(prior)):
                if u == 0:
                    series_loss = my_kl_loss(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.args.win_size)).detach()) * temperature
                    prior_loss = my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.args.win_size)),
                        series[u].detach()) * temperature
                else:
                    series_loss += my_kl_loss(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.args.win_size)).detach()) * temperature
                    prior_loss += my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.args.win_size)),
                        series[u].detach()) * temperature
            metric = torch.softmax((-series_loss - prior_loss), dim=-1)
            if self.model.name == "DCDetector":
                cri = metric.detach().cpu().numpy()
            else: 
                cri = (metric*loss).detach().cpu().numpy()
            attens_energy_test.append(cri)
        attens_energy_test = np.concatenate(attens_energy_test, axis=0).reshape(-1)
        test_energy = np.array(attens_energy_test)
        test_labels = np.concatenate(test_labels, axis=0).reshape(-1)
        return test_energy, test_labels
    def opt_anomaly(self, setting):
        _,test_loader = self._get_data("test")
        _,thre_loader = self._get_data("threshold")
        _,train_loader = self._get_data("train")
        list_pred_models = []
        list_thresholds = []

        model_path = os.path.join("./checkpoints/",setting)
        model_list = [checkpoint for checkpoint in sorted(os.listdir(model_path))]

        for index in trange(len(model_list), desc=f'[Opt Anomaly]'):
            if model_list[index].split("checkpoint_")[1].split(".")[0].find("slow_learner") != -1:
                model_name = model_list[index].split("checkpoint_")[1].split(".")[0].split("slow_learner_")[1]
                self.model = self.model_dict[model_name].Model(self.args).float().to(self.device)

                model_load_state = torch.load(os.path.join("./checkpoints/",setting,model_list[index]))
                self.model.load_state_dict(model_load_state)
                temperature = 50 #For Association discrepancy

                self.model.eval()
                # (1) stastic on the TRAIN SET
                train_energy = self.calculate_train_energy(train_loader,temperature,slow_learner=True)
                # (2) stastic on the TEST SET
                test_energy, test_labels = self.calculate_test_energy(test_loader,temperature,slow_learner=True)
                # (3) stastic on the THRESHOLD SET
                threshold_energy, _ = self.calculate_test_energy(thre_loader,temperature,slow_learner=True)
                combined_energy = np.concatenate([train_energy, threshold_energy], axis=0)
                threshold = np.percentile(combined_energy, 100 - self.args.anomaly_ratio)
                print(f"Threshold SLOW LEARNER {model_name}: {threshold}")

                pred = (test_energy > threshold).astype(int)
                gt = test_labels.astype(int)
                gt, pred = adjustment(gt, pred)
                accuracy = accuracy_score(gt, pred)
                precision, recall, f_score, support = precision_recall_fscore_support(gt, pred,
                                                                                    average='binary')
                print("Model Name:{}, Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(
                        model_name, accuracy, precision,
                        recall, f_score))
                
                list_pred_models.append(test_energy)
                list_thresholds.append(threshold)
            else:
                model_name = model_list[index].split("checkpoint_")[1].split(".")[0]
                self.model = self.model_dict[model_name].Model(self.args).float().to(self.device)

                model_load_state = torch.load(os.path.join("./checkpoints/",setting,model_list[index]))
                self.model.load_state_dict(model_load_state)
                temperature = 50 #For Association discrepancy

                self.model.eval()
                # (1) stastic on the TRAIN SET
                train_energy = self.calculate_train_energy(train_loader,temperature)
                # (2) stastic on the TEST SET
                test_energy, test_labels = self.calculate_test_energy(test_loader,temperature)
                # (3) stastic on the THRESHOLD SET
                threshold_energy, _ = self.calculate_test_energy(thre_loader,temperature)

                combined_energy = np.concatenate([train_energy, threshold_energy], axis=0)
                threshold = np.percentile(combined_energy, 100 - self.args.anomaly_ratio)
                print(f"Threshold NORMAL LEARNER {model_name}: {threshold}")

                pred = (test_energy > threshold).astype(int)
                gt = test_labels.astype(int)
                gt, pred = adjustment(gt, pred)

                print(f"pred: {len(pred)}")
                print(f"gt: {len(gt)}")
                accuracy = accuracy_score(gt, pred)
                precision, recall, f_score, support = precision_recall_fscore_support(gt, pred,
                                                                       average='binary')
                print("Model Name:{}, Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(
                        model_name, accuracy, precision,
                        recall, f_score))
                
                print(f"Total number of reported anomalies before RL {model_name}: {sum(pred)}")
                print(f"Total number of true anomalies {model_name}: {sum(gt)}") 

                list_pred_models.append(test_energy)
                list_thresholds.append(threshold)
                
        f_csv = open("training_anomaly_detection_asso_discrep_rl.csv","a")
        csvreader = csv.writer(f_csv)
        header = [[setting],["Precision RL","Recall RL","F1-score RL","Reward RL"]]
        csvreader.writerows(header)

        logdir = "logs"
        path = os.path.join(logdir, setting)
        if not os.path.exists(path):
            os.makedirs(path)
        # Set up the training environment on all the dataset
        env_off=TrainEnvOffline_dist_conf(list_pred_sc=list_pred_models, list_thresholds=list_thresholds, list_gtruth=test_labels)
        # Train the model on all the dataset  
        model = DQN('MlpPolicy', env_off, verbose=1, tensorboard_log=path)
        model.learn(total_timesteps=len(list_pred_models[0]),progress_bar=True) 
        acc,prec, rec, f1, _, list_preds, reward =eval_model(model, env_off)  #masuk ke step di env

        print(f"precision: {prec}, recall: {rec}, F1-score: {f1} reward: {reward}")
        csvreader.writerows([[prec,rec,f1,reward]])

        print("Total number of reported anomalies: ",sum(list_preds))
        print("Total number of true anomalies: ",sum(test_labels)) #di RL ambil 10000 time step aja

        print("accuracy: %.4f" % (acc))
        print("precision: %.4f" % (prec))
        print("recall: %.4f" % (rec))
        print("F1-score: %.4f" % (f1))
        return
    