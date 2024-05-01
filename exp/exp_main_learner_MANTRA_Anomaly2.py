from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping,EarlyStopping_Slow_Learner, adjust_learning_rate, adjustment, my_kl_loss
from utils.pot import pot_eval
from utils.diagnosis import ndcg, hit_att
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
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


class Exp_Anomaly_Detection(Exp_Basic):
    def __init__(self, args):
        super(Exp_Anomaly_Detection, self).__init__(args)

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
    
    def _select_slow_optimizer(self):
        slow_model_optim = optim.Adam(self.slow_model.parameters(), lr=self.args.learning_rate)
        return slow_model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali_fast_learner(self, vali_data, vali_loader, criterion):
        total_loss = []
        # loss1
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, _) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)

                if self.model.name not in ["KBJNet"]:
                    outputs_fast_learner_vali = self.model(batch_x)
                else:
                    outputs_fast_learner_vali = self.model(batch_x.permute(0,2,1)) 

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs_fast_learner_vali = outputs_fast_learner_vali[:, :, f_dim:]

                pred = outputs_fast_learner_vali.detach().cpu()
                true = batch_x.detach().cpu()

                loss = criterion(pred, true)
                total_loss.append(loss)
                
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss, outputs_fast_learner_vali
    def vali_slow_learner(self, valid_data, vali_loader, outputs_fast_learner):
        self.slow_model.eval()
        loss_1 = []
        loss_2 = []
        # Anomaly Transformer doesnt use NO GRAD in vali
        with torch.no_grad():
            for i, (batch_x, _) in enumerate(vali_loader):
                s0,s1,s2 = batch_x.shape
                randuniform = torch.empty(s0,s1,s2).uniform_(0, 1)
                m_ones = torch.ones(s0,s1,s2)
                slow_mark = torch.bernoulli(randuniform)
                batch_x_slow = batch_x.clone()
                batch_x_slow = batch_x_slow * (m_ones-slow_mark).to(self.device)
                if self.slow_model.name  == "KBJNet":
                    _ = self.slow_model.forward(batch_x_slow)
                elif self.slow_model.name == "MaelNetS2":
                    _ , [series,prior] = self.slow_model.forward(batch_x_slow.permute(0,2,1))
                else:
                    _ = self.slow_model.forward(batch_x_slow.permute(0,2,1))
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

                rec_loss = self.criterion(outputs_fast_learner, batch_x.float().to(self.device))
                loss_1.append((rec_loss - self.args.k * series_loss).item())
                loss_2.append((rec_loss + self.args.k * prior_loss).item())

        self.slow_model.train()
        return np.average(loss_1), np.average(loss_2)
    
    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)

        early_stopping_fast_learner = EarlyStopping(patience=self.args.patience, verbose=True)
        early_stopping_slow_learner = EarlyStopping_Slow_Learner(patience=self.args.patience, verbose=True, slow_learner=True)

        model_optim = self._select_optimizer()
        slow_model_optim = self._select_slow_optimizer()
        
        criterion = self._select_criterion()

        f = open("training_anomaly_detection.txt", 'a')
        f_csv = open("training_anomaly_detection.csv","a")

        csvreader = csv.writer(f_csv)
        for epoch in tqdm(list(range(self.args.train_epochs))):
            iter_count = 0
            iter_count_slow = 0

            train_fast_learner_loss = []
            train_slow_learner_loss = []

            self.model.train()
            self.slow_model.train()

            epoch_time = time.time()
            for i, (batch_x, _) in enumerate(train_loader):
                #  ========== FAST LEARNER ============
                if not early_stopping_fast_learner.early_stop:
                    iter_count += 1
                    model_optim.zero_grad()

                    batch_x = batch_x.float().to(self.device)
                    if self.model.name not in ["KBJNet"]:
                        outputs = self.model(batch_x)
                    else:
                        outputs = self.model(batch_x.permute(0,2,1)) 
                    
                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, :, f_dim:]

                    rec_loss = criterion(outputs, batch_x)
                    train_fast_learner_loss.append((rec_loss - self.args.k * series_loss).item())
                    if (i + 1) % 100 == 0:
                        print("\titers FAST LEARNER : {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, rec_loss.item()))
                        speed = (time.time() - time_now) / iter_count
                        left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                        print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                        iter_count = 0
                        time_now = time.time()
                    rec_loss.backward()
                    model_optim.step()

                # ========== SLOW LEARNER ============
                if not early_stopping_slow_learner.early_stop:
                    iter_count_slow = iter_count_slow + 1
                    slow_model_optim.zero_grad() 
                    s0,s1,s2 = batch_x.shape
                    randuniform = torch.empty(s0,s1,s2).uniform_(0, 1)
                    m_ones = torch.ones(s0,s1,s2)
                    slow_mark = torch.bernoulli(randuniform)
                    batch_x_slow = batch_x.clone()
                    batch_x_slow = batch_x_slow * (m_ones-slow_mark).to(self.device)

                    if self.slow_model.name  == "KBJNet":
                        _ = self.slow_model.forward(batch_x_slow)
                    elif self.slow_model.name == "MaelNetS2":
                        _ , [series,prior] = self.slow_model.forward(batch_x_slow.permute(0,2,1))
                    else:
                        _ = self.slow_model.forward(batch_x_slow.permute(0,2,1))

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

                    train_slow_learner_loss.append((rec_loss - self.args.k * series_loss).item())
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
                    self.slow_model.step()
            #  ========== FAST LEARNER ============
            if not early_stopping_fast_learner.early_stop:
                train_fast_learner_loss = np.average(train_fast_learner_loss)
                vali_fast_learner_loss, outputs_fast_learner_vali = self.vali_fast_learner(vali_data, vali_loader, criterion)
                test_fast_learner_loss, outputs_fast_learner_test = self.vali_fast_learner(test_data, test_loader, criterion)
                data_for_csv = [[epoch + 1, time.time() - epoch_time, train_steps, round(train_fast_learner_loss,7), round(vali_fast_learner_loss,7), round(test_fast_learner_loss,7)],[]]
                print("FAST LEARNER Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                    epoch + 1, train_steps, train_fast_learner_loss, vali_fast_learner_loss, test_fast_learner_loss))
                f.write("FAST LEARNER Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                    epoch + 1, train_steps, train_fast_learner_loss, vali_fast_learner_loss, test_fast_learner_loss))
                f.write("\n")
                csvreader.writerows(data_for_csv)
                adjust_learning_rate(model_optim, epoch + 1, self.args)
            #  ========== SLOW LEARNER ============
            if not early_stopping_slow_learner.early_stop:
                train_slow_loss = np.average(train_slow_learner_loss)
                vali_slow_loss1, vali_slow_loss2 = self.vali_slow_learner(vali_data, vali_loader, outputs_fast_learner_vali)
                test_slow_loss1, test_slow_loss2 = self.vali_slow_learner(test_data, test_loader, outputs_fast_learner_test)
                data_for_csv = [[epoch + 1, time.time() - epoch_time, train_steps, round(train_slow_loss,7), round(vali_slow_loss1,7), round(test_slow_loss1,7)],[]]
                print("SLOW LEARNER Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                    epoch + 1, train_steps, train_slow_loss, vali_slow_loss1, test_slow_loss1))
                f.write("SLOW LEARNER Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                    epoch + 1, train_steps, train_slow_loss, vali_slow_loss1, test_slow_loss1))
                f.write("\n")
                csvreader.writerows(data_for_csv)
                adjust_learning_rate(slow_model_optim, epoch + 1, self.args)
            if epoch == 0:
                f.write(setting + "  \n")
                header = [[setting],["Epoch","Cost Time", "Steps", "Train Loss", "Vali Loss", "Test Loss"]]
                csvreader.writerows(header)

            # Saving Model
            early_stopping_fast_learner(vali_fast_learner_loss, self.model, path)
            early_stopping_slow_learner(vali_slow_loss1, vali_slow_loss2, self.slow_model, path)

            if early_stopping_fast_learner.early_stop and early_stopping_slow_learner.early_stop:
                print("Early stopping")
                break
        csvreader.writerow([])
        return

    def test(self, setting, test=1):
        _, test_loader = self._get_data(flag='test')
        _, train_loader = self._get_data(flag='train')

        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))
            self.slow_model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint_slow_learner.pth')))

        attens_energy = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        self.slow_model.eval()

        self.anomaly_criterion = nn.MSELoss(reduce=False)
        temperature = 50
        # (1) stastic on the TRAIN SET
        #  ========== FAST LEARNER ============
        with torch.no_grad():
            for i, (batch_x, _) in enumerate(train_loader):
                batch_x = batch_x.float().to(self.device)
                # reconstruction
                if self.model.name not in ["KBJNet"]:
                    outputs = self.model(batch_x)
                else:
                    outputs = self.model(batch_x.permute(0,2,1)) 
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, :, f_dim:]
                # criterion
                loss = self.anomaly_criterion(batch_x, outputs)
                score = torch.mean(loss, dim=-1)

        #  ========== SLOW LEARNER ============
                s0,s1,s2 = batch_x.shape
                randuniform = torch.empty(s0,s1,s2).uniform_(0, 1)
                m_ones = torch.ones(s0,s1,s2)
                slow_mark = torch.bernoulli(randuniform)
                batch_x_slow = batch_x.clone()
                batch_x_slow = batch_x_slow * (m_ones-slow_mark).to(self.device)

                if self.slow_model.name  == "KBJNet":
                    _ = self.slow_model.forward(batch_x_slow)
                elif self.slow_model.name == "MaelNetS2":
                    _ , [series,prior] = self.slow_model.forward(batch_x_slow.permute(0,2,1))
                else:
                    _ = self.slow_model.forward(batch_x_slow.permute(0,2,1))

                series_loss = 0.0
                prior_loss = 0.0
                for u in range(len(prior)):
                    if u == 0:
                        series_loss = my_kl_loss(series[u], (
                                prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                    self.win_size)).detach()) * temperature
                        prior_loss = my_kl_loss(
                            (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                    self.win_size)),
                            series[u].detach()) * temperature
                    else:
                        series_loss += my_kl_loss(series[u], (
                                prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                    self.win_size)).detach()) * temperature
                        prior_loss += my_kl_loss(
                            (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                    self.win_size)),
                            series[u].detach()) * temperature
                metric = torch.softmax((-series_loss - prior_loss), dim=-1)
                cri = metric * score
                cri = cri.detach().cpu().numpy()
                attens_energy.append(cri)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        train_energy = np.array(attens_energy)

        # (2) find the threshold TEST SET
        attens_energy = []
        test_labels = []
        # (1) stastic on the TRAIN SET
        #  ========== FAST LEARNER ============
        with torch.no_grad():
            for i, (batch_x, labels) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                # reconstruction
                if self.model.name not in ["KBJNet"]:
                    outputs = self.model(batch_x)
                else:
                    outputs = self.model(batch_x.permute(0,2,1)) 
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, :, f_dim:]
                # criterion
                loss = self.anomaly_criterion(batch_x, outputs)
                score = torch.mean(loss, dim=-1)

        #  ========== SLOW LEARNER ============
                s0,s1,s2 = batch_x.shape
                randuniform = torch.empty(s0,s1,s2).uniform_(0, 1)
                m_ones = torch.ones(s0,s1,s2)
                slow_mark = torch.bernoulli(randuniform)
                batch_x_slow = batch_x.clone()
                batch_x_slow = batch_x_slow * (m_ones-slow_mark).to(self.device)

                if self.slow_model.name  == "KBJNet":
                    _ = self.slow_model.forward(batch_x_slow)
                elif self.slow_model.name == "MaelNetS2":
                    _ , [series,prior] = self.slow_model.forward(batch_x_slow.permute(0,2,1))
                else:
                    _ = self.slow_model.forward(batch_x_slow.permute(0,2,1))

                series_loss = 0.0
                prior_loss = 0.0
                for u in range(len(prior)):
                    if u == 0:
                        series_loss = my_kl_loss(series[u], (
                                prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                    self.win_size)).detach()) * temperature
                        prior_loss = my_kl_loss(
                            (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                    self.win_size)),
                            series[u].detach()) * temperature
                    else:
                        series_loss += my_kl_loss(series[u], (
                                prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                    self.win_size)).detach()) * temperature
                        prior_loss += my_kl_loss(
                            (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                    self.win_size)),
                            series[u].detach()) * temperature
                metric = torch.softmax((-series_loss - prior_loss), dim=-1)
                cri = metric * score
                cri = cri.detach().cpu().numpy()
                attens_energy.append(cri)
                test_labels.append(labels)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)
        combined_energy = np.concatenate([train_energy, test_energy], axis=0)

        threshold = np.percentile(combined_energy, 100 - self.args.anomaly_ratio)

        print("Threshold :", threshold)

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
        print("pred: ", pred.shape)
        print("gt:   ", gt.shape)

        accuracy = accuracy_score(gt, pred)
        precision, recall, f_score, support = precision_recall_fscore_support(gt, pred, average='binary')
        print("Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(
            accuracy, precision,
            recall, f_score))

        # result_anomaly_detection.txt
        f = open("result_anomaly_detection.txt", 'a')
        f.write(setting + "  \n")
        f.write("Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(
            accuracy, precision,
            recall, f_score))
        f.write('\n')
        f.write('\n')
        f.close()
        #result_anomaly_detection.csv
        f_csv = open("result_anomaly_detection.csv","a")
        csvreader = csv.writer(f_csv)
        datas = [[setting],["Accuracy","Precision","Recall","F-score"],[round(accuracy,4),round(precision,4),round(recall,4),round(f_score,4)]]
        csvreader.writerows(datas)
        return