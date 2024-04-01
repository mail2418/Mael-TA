from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, adjustment
from utils.slowloss import ssl_loss_v2
from utils.metrics import NegativeCorr
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
import torch.multiprocessing
from tqdm import tqdm
from pprint import pprint
import csv
from collections import defaultdict
torch.multiprocessing.set_sharing_strategy('file_system')
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
from typing import List
from utils.agentreward import TrainEnvOffline_dist_conf, eval_model
from stable_baselines3 import DQN
warnings.filterwarnings('ignore')


class Exp_Anomaly_Detection_Learner(Exp_Basic):
    def __init__(self, args):
        super(Exp_Anomaly_Detection_Learner, self).__init__(args)
        self.anomaly_criterion = nn.MSELoss(reduce=False)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model
    def _build_slow_model(self):
        slow_model = self.model_dict[self.args.slow_model].Model(self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            slow_model = nn.DataParallel(slow_model, device_ids=self.args.device_ids)
        return slow_model
    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader
    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim
    def _select_criterion(self):
        if self.args.loss_type == "neg_corr":
            criterion = NegativeCorr(self.args.correlation_penalty)
        else:
            criterion = nn.MSELoss()
        return criterion
    def vali(self, vali_loader, criterion, epoch, flag):
        steps = len(vali_loader)
        iter_count = 0
        time_now = time.time()
        total_loss = []
        with torch.no_grad():
            for i , (batch_x, batch_y) in enumerate(vali_loader):
                if i == 200: break
                iter_count = iter_count + 1
                batch_x = batch_x.float().to(self.device)
                f_dim = -1 if self.args.features == 'MS' else 0
                if self.model.name not in ["KBJNet"]:
                    outputs = self.model(batch_x)
                else:
                    outputs = self.model(batch_x.permute(0,2,1)) 
                outputs = outputs[:, :, f_dim:]
                pred = outputs.detach().cpu()
                true = batch_x.detach().cpu()
                loss = criterion(pred, true)
                total_loss.append(loss.item())
                if (i + 1) % 100 == 0:
                    print("\titers {3}: {0}, epoch : {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item(), flag))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        path_ds = os.path.join(self.args.root_path, setting) #Path data for RL
        if not os.path.exists(path_ds):
            os.makedirs(path_ds)
        time_now = time.time()

        train_steps = len(train_loader)      
        f = open("training_mantra_anomaly_detection.txt", 'a')
        f_csv = open("training_mantra_anomaly_detection.csv","a")
        csvreader = csv.writer(f_csv)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()  
        for epoch in tqdm(list(range(self.args.train_epochs))):
            iter_count = 0
            train_loss = []
            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y) in enumerate(train_loader):
                if i == 200: break
                iter_count = iter_count + 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                f_dim = -1 if self.args.features == 'MS' else 0
                if self.model.name not in ["KBJNet"]:
                    outputs = self.model(batch_x)
                else:
                    outputs = self.model(batch_x.permute(0,2,1)) 
                outputs = outputs[:, :, f_dim:].detach().cpu()
                loss = criterion(outputs, batch_x.detach().cpu())
                train_loss.append(loss.item())           
                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()
                # slow_model_optim.zero_grad()    
                loss.backward()
                # slow_model_optim.step()
                model_optim.step()

            if epoch == 0:
                f.write(setting + "  \n")    
                # f.write(setting + f"learner_{idx+1}  \n")    
                header = [[setting],["Epoch","Cost Time", "Steps", "Train Loss", "Vali Loss", "Test Loss"]]
                csvreader.writerows(header)
            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            f.write("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            f.write("\n")
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_loader, criterion, epoch, "Validation")
            test_loss = self.vali(test_loader, criterion, epoch, "Test")

            data_for_csv = [[epoch + 1, time.time() - epoch_time, train_steps, round(train_loss,7), round(vali_loss,7), round(test_loss,7)],[]]
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            f.write("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            f.write("\n")
            csvreader.writerows(data_for_csv)

            # Saving Model
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            adjust_learning_rate(model_optim, epoch + 1, self.args)
        f.write("\n")
        csvreader.writerow([])
        f.close()

    def test(self, setting, test=1):
        _, test_loader = self._get_data(flag='test')
        _, train_loader = self._get_data(flag='train')

        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        list_gtruth = []
        list_thresholds = []
        list_predsrc = []
        for learner_idx in range(self.args.n_learner):
            attens_energy = []
            self.model.eval()
            self.anomaly_criterion = nn.MSELoss(reduce=False)
            # (1) stastic on the TRAIN SET
            with torch.no_grad():
                for i, (batch_x, _) in enumerate(train_loader):
                    if i == 200: break
                    
                    batch_x = batch_x.float().to(self.device)
                    # reconstruction
                    if self.model.name not in ["KBJNet"]:
                        outputs = self.model.forward_1learner(batch_x,learner_idx)
                    else:
                        outputs = self.model.forward_1learner(batch_x.permute(0,2,1),learner_idx) 
                    # Slow Learner
                    s0,s1,s2 = batch_x.shape
                    randuniform = torch.empty(s0,s1,s2).uniform_(0, 1)
                    m_ones = torch.ones(s0,s1,s2)
                    slow_mark = torch.bernoulli(randuniform)
                    batch_x_slow = batch_x.clone()
                    batch_x_slow = batch_x_slow * (m_ones-slow_mark).to(self.device)

                    if self.slow_model.name not in ["KBJNet"]:
                        slow_out = self.slow_model.forward(batch_x_slow)
                    else:
                        slow_out = self.slow_model.forward(batch_x_slow.permute(0,2,1))

                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, :, f_dim:]
                    # criterion
                    loss = self.anomaly_criterion(batch_x, outputs)
                    loss = loss + ssl_loss_v2(slow_out, batch_x, slow_mark, s1, s2, self.device)
                    score = torch.mean(loss, dim=-1)
                    score = score.detach().cpu().numpy()
                    attens_energy.append(score)

            attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
            train_energy = np.array(attens_energy)

            # (2) find the threshold TEST SET
            attens_energy = []
            test_labels = []
            for i, (batch_x, batch_y) in enumerate(test_loader):
                if i == 200: break
                batch_x = batch_x.float().to(self.device)
                # reconstruction
                if self.model.name not in ["KBJNet"]:
                    outputs = self.model.forward_1learner(batch_x,learner_idx)
                else:
                    outputs = self.model.forward_1learner(batch_x.permute(0,2,1),learner_idx) 

                # Slow Learner
                s0,s1,s2 = batch_x.shape
                randuniform = torch.empty(s0,s1,s2).uniform_(0, 1)
                m_ones = torch.ones(s0,s1,s2)
                slow_mark = torch.bernoulli(randuniform)
                batch_x_slow = batch_x.clone()
                batch_x_slow = batch_x_slow * (m_ones-slow_mark).to(self.device)

                if self.slow_model.name not in ["KBJNet"]:
                    slow_out = self.slow_model.forward(batch_x_slow)
                else:
                    slow_out = self.slow_model.forward(batch_x_slow.permute(0,2,1))

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, :, f_dim:]
                # criterion
                lossT = self.anomaly_criterion(batch_x, outputs)
                loss = loss + ssl_loss_v2(slow_out, batch_x, slow_mark, s1, s2, self.device)
                score = torch.mean(lossT, dim=-1)
                score = score.detach().cpu().numpy()
                attens_energy.append(score)
                test_labels.append(batch_y)

            attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
            test_energy = np.array(attens_energy)
            combined_energy = np.concatenate([train_energy, test_energy], axis=0)

            threshold = np.percentile(combined_energy, 100 - self.args.anomaly_ratio)

            print("Threshold :", threshold)
            list_thresholds.append(threshold)
            # (3) evaluation on the test set
            pred = (test_energy > threshold).astype(int)
            test_labels = np.concatenate(test_labels, axis=0).reshape(-1)
            test_labels = np.array(test_labels)

            gt = test_labels.astype(int)
            print("pred:   ", pred.shape)
            print("gt:     ", gt.shape)

            # (4) detection adjustment
            gt, pred = adjustment(gt, pred) #gt == label

            pred = np.array(pred)
            gt = np.array(gt)
            print(f"Ground Truth and Prediction of Learner {learner_idx + 1}")
            print("pred: ", pred.shape)
            print("gt:   ", gt.shape)
            list_predsrc.append(pred)
            if learner_idx == self.args.n_learner - 1:
                list_gtruth.append(gt)

        EXP_TIMES=10 # How many runs to average the results
        # Store the precision, recall, F1-score
        store_prec=np.zeros(EXP_TIMES)
        store_rec=np.zeros(EXP_TIMES)
        store_f1=np.zeros(EXP_TIMES)

        for times in range(EXP_TIMES):
            # Set up the training environment on all the dataset
            # env_off=TrainEnvOffline(list_pred_sc=list_pred_sc, list_thresholds=list_thresholds, list_gtruth=list_gtruth)
            # env_off=TrainEnvOffline_consensus_conf(list_pred_sc=list_pred_sc, list_thresholds=list_thresholds, list_gtruth=list_gtruth)
            env_off=TrainEnvOffline_dist_conf(list_pred_sc=list_predsrc, list_thresholds=list_thresholds, list_gtruth=list_gtruth)

            # Train the model on all the dataset  
            model = DQN('MlpPolicy', env_off, verbose=0)
            model.learn(total_timesteps=len(list_predsrc[0])) 
            # model.save("DQN_offline_model")
            # model.save("A2C_offline_model")
            
            # Evaluate the model on all the dataset
            # model = DQN.load("DQN_offline_model")
            # model=A2C.load("A2C_offline_model")
            prec, rec, f1, _, list_preds=eval_model(model, env_off)  #masuk ke step di env

            store_prec[times]=prec
            store_rec[times]=rec
            store_f1[times]=f1

        # Compute the mean and standard deviation of the results
        average_prec=np.mean(store_prec)
        average_rec=np.mean(store_rec)
        average_f1=np.mean(store_f1)

        std_prec=np.std(store_prec)
        std_rec=np.std(store_rec)
        std_f1=np.std(store_f1)

        print("Total number of reported anomalies: ",sum(list_preds))
        print("Total number of true anomalies: ",sum(list_gtruth))

        print("Average precision: %.4f, std: %.4f" % (average_prec, std_prec))
        print("Average recall: %.4f, std: %.4f" % (average_rec, std_rec))
        print("Average F1-score: %.4f, std: %.4f" % (average_f1, std_f1))
        return