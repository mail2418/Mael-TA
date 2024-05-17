from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping_Asso_Discrep, adjust_learning_rate, adjustment, my_kl_loss
from utils.slowloss import ssl_loss_v2
import torch.multiprocessing
from tqdm import tqdm
from pprint import pprint
torch.multiprocessing.set_sharing_strategy('file_system')
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
import csv
warnings.filterwarnings('ignore')

class Exp_Anomaly_Detection_SL(Exp_Basic):
    def __init__(self, args):
        super(Exp_Anomaly_Detection_SL, self).__init__(args)
    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model
    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader
    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim
    def vali(self, vali_loader):
        self.model.eval()
        loss_1 = []
        loss_2 = []
        # Anomaly Transformer doesnt use NO GRAD in vali
        with torch.no_grad():
            for i, (batch_x, _) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                s0,s1,s2 = batch_x.shape
                randuniform = torch.empty(s0,s1,s2).uniform_(0, 1)
                m_ones = torch.ones(s0,s1,s2)
                slow_mark = torch.bernoulli(randuniform)
                batch_x_slow = batch_x.clone()
                batch_x_slow = batch_x_slow * (m_ones-slow_mark).to(self.device)
                outputs, series, prior= self.model(batch_x_slow)
                series_loss = 0.0
                prior_loss = 0.0
                for u in range(len(prior)):
                    series_loss += (torch.mean(my_kl_loss(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.win_size)).detach())) + torch.mean(
                        my_kl_loss(
                            (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                    self.win_size)).detach(),
                            series[u])))
                    prior_loss += (torch.mean(
                        my_kl_loss((prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                        self.win_size)),
                                series[u].detach())) + torch.mean(
                        my_kl_loss(series[u].detach(),
                                (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                       self.win_size)))))
                series_loss = series_loss / len(prior)
                prior_loss = prior_loss / len(prior)
                rec_loss = ssl_loss_v2(outputs, batch_x_slow, slow_mark, s1, s2, self.device)
                # rec_loss = criterion(outputs, batch_x.float().to(self.device))
                loss_1.append((rec_loss - self.args.k * series_loss).item())
                loss_2.append((rec_loss + self.args.k * prior_loss).item())
        self.model.train()
        return np.average(loss_1), np.average(loss_2)
    
    def train(self, setting):
        _, train_loader = self._get_data(flag='train')
        _, vali_loader = self._get_data(flag='val')
        _, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)
        time_now = time.time()
        train_steps = len(train_loader)
        early_stopping = EarlyStopping_Asso_Discrep(patience=self.args.patience, verbose=True, slow_learner=True)
        model_optim = self._select_optimizer()    

        f = open("training_anomaly_detection_asso_discrep_slow.txt", 'a')
        f_csv = open("training_anomaly_detection_asso_discrep_slow.csv","a")

        csvreader = csv.writer(f_csv)
        for epoch in tqdm(list(range(self.args.train_epochs))):
            iter_count = 0
            train_loss = []
            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, _) in enumerate(train_loader):
                iter_count = iter_count + 1
                batch_x = batch_x.float().to(self.device)
                model_optim.zero_grad() 

                s0,s1,s2 = batch_x.shape
                randuniform = torch.empty(s0,s1,s2).uniform_(0, 1)
                m_ones = torch.ones(s0,s1,s2)
                slow_mark = torch.bernoulli(randuniform)
                batch_x_slow = batch_x.clone()
                batch_x_slow = batch_x_slow * (m_ones-slow_mark).to(self.device)

                outputs, series, prior = self.model.forward(batch_x_slow)

                series_loss = 0.0
                prior_loss = 0.0
                for u in range(len(prior)):
                    series_loss += (torch.mean(my_kl_loss(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.win_size)).detach())) + torch.mean(
                        my_kl_loss((prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                        self.win_size)).detach(),
                                series[u])))
                    prior_loss += (torch.mean(my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.win_size)),
                        series[u].detach())) + torch.mean(
                        my_kl_loss(series[u].detach(), (
                                prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                    self.win_size)))))
                series_loss = series_loss / len(prior)
                prior_loss = prior_loss / len(prior)

                rec_loss = ssl_loss_v2(outputs, batch_x_slow, slow_mark, s1, s2, self.device)
                train_loss.append((rec_loss - self.args.k * series_loss).item())
                loss1 = rec_loss - self.args.k * series_loss # minimise phase
                loss2 = rec_loss + self.args.k * prior_loss # maximise phase

                if (i + 1) % 100 == 0:
                    print("\titers SLOW LEARNER: {0}, epoch: {1} | loss minimise phase: {2:.7f} | loss maximise phase: {2:.7f}".format(i + 1, epoch + 1, loss1.item(), loss2.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss1.backward(retrain_graph=True)
                loss2.backward()
                self.model.step()
            train_slow_loss = np.average(train_loss)
            vali_slow_loss1, vali_slow_loss2 = self.vali(vali_loader)
            test_slow_loss1, test_slow_loss2 = self.vali(test_loader)

            if epoch == 0:
                f.write(setting + "  \n")
                header = [[setting],["Epoch","Cost Time", "Steps", "Train Loss", "Vali Loss", "Test Loss"]]
                csvreader.writerows(header)
            data_for_csv = [[epoch + 1, time.time() - epoch_time, train_steps, round(train_slow_loss,7), round(vali_slow_loss1,7), round(test_slow_loss1,7)],[]]
            print("SLOW LEARNER Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_slow_loss, vali_slow_loss1, test_slow_loss1))
            f.write("SLOW LEARNER Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_slow_loss, vali_slow_loss1, test_slow_loss1))
            f.write("\n")
            csvreader.writerows(data_for_csv)
            # Saving Model
            early_stopping(vali_slow_loss1, vali_slow_loss2, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            adjust_learning_rate(model_optim, epoch + 1, self.args)
        csvreader.writerow([])
        print(f"Train is finished for slow model {self.model.name}")
        return