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
warnings.filterwarnings('ignore')


class Exp_Anomaly_Detection_Learner(Exp_Basic):
    def __init__(self, args):
        super(Exp_Anomaly_Detection_Learner, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()
        self.slow_model = self.model_dict[self.args.slow_model].Model(self.args).float().to(self.device)
        
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

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
        if self.args.loss_type == "neg_corr":
            criterion = NegativeCorr(self.args.correlation_penalty)
        else:
            criterion = nn.MSELoss()
        return criterion
    
    def vali_step_validator(self, vali_loader, criterion, header_bmnpzreader, valid_X:List, epoch, flag):
        total_loss = []
        all_bm_valid_test_outputs = [[] for _ in range(self.args.n_learner)]
        with torch.no_grad():
            for i , (batch_x, _) in enumerate(vali_loader):
                if epoch == 0:
                    valid_X.extend(batch_x.detach().cpu().numpy())

                batch_x = batch_x.float().to(self.device)

                list_of_bm_valid_test = [[] for _ in range(self.args.n_learner)]
                dec_out = []
                f_dim = -1 if self.args.features == 'MS' else 0
                for idx in range(self.args.n_learner):
                    if self.model.name not in ["KBJNet"]:
                        outputs = self.model.forward_1learner(batch_x, idx)
                    else:
                        outputs = self.model.forward_1learner(batch_x.permute(0,2,1),idx) 
                    list_of_bm_valid_test[idx].append(outputs)
                    dec_out.append(outputs)

                for idx in range(self.args.n_learner):
                    list_of_bm_valid_test[idx] = torch.stack(list_of_bm_valid_test[idx])
                    list_of_bm_valid_test[idx] = torch.mean(list_of_bm_valid_test[idx],axis=0)
                    list_of_bm_valid_test[idx] = list_of_bm_valid_test[idx][:,:,f_dim:]
                    list_of_bm_valid_test[idx] = list_of_bm_valid_test[idx].detach().cpu().numpy().reshape(list_of_bm_valid_test[idx].shape[0] * list_of_bm_valid_test[idx].shape[1],-1)

                    all_bm_valid_test_outputs[idx].extend(list_of_bm_valid_test[idx])
                outputs = torch.stack(dec_out)
                outputs = torch.mean(outputs,axis=0) 
                outputs = outputs[:, :, f_dim:]

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

                pred = outputs.detach().cpu()
                true = batch_x.detach().cpu()

                loss = criterion(pred, true) + ssl_loss_v2(slow_out, batch_x, slow_mark, s1, s2, self.device)
                total_loss.append(loss.item())
                if i == 300: break
        dict_learner_valid = {}
        for learner_idx in range(self.args.n_learner):
            dict_learner_valid[header_bmnpzreader[learner_idx]] = np.array(all_bm_valid_test_outputs[learner_idx], dtype="object")
        if epoch == 0:
            # np_input_valid_X = valid_X[0]
            # for i in range(1,valid_X[0].shape[0]):
            #     np_input_valid_X = np.append(np_input_valid_X,valid_X[i])
            np_input_valid_X = np.array(valid_X, dtype="object")
            np.save(f"{self.args.root_path}valid_X.npy",np_input_valid_X)
        total_loss = np.average(total_loss)
        return total_loss,dict_learner_valid
    
    def vali_step_tester(self, vali_loader, criterion, header_bmnpzreader, test_X:List,test_y:List, epoch, flag):
        total_loss = []
        all_bm_valid_test_outputs = [[] for _ in range(self.args.n_learner)]
        with torch.no_grad():
            for i , (batch_x, batch_y) in enumerate(vali_loader):
                if epoch == 0:
                    test_X.extend(batch_x.detach().cpu().numpy())
                    test_y.extend(batch_y.detach().cpu().numpy())
                batch_x = batch_x.float().to(self.device)

                list_of_bm_valid_test = [[] for _ in range(self.args.n_learner)]
                dec_out = []
                f_dim = -1 if self.args.features == 'MS' else 0
                for idx in range(self.args.n_learner):
                    if self.model.name not in ["KBJNet"]:
                        outputs = self.model.forward_1learner(batch_x, idx)
                    else:
                        outputs = self.model.forward_1learner(batch_x.permute(0,2,1),idx) 
                    list_of_bm_valid_test[idx].append(outputs)
                    dec_out.append(outputs)

                for idx in range(self.args.n_learner):
                    list_of_bm_valid_test[idx] = torch.stack(list_of_bm_valid_test[idx])
                    list_of_bm_valid_test[idx] = torch.mean(list_of_bm_valid_test[idx],axis=0)
                    list_of_bm_valid_test[idx] = list_of_bm_valid_test[idx][:,:,f_dim:]
                    list_of_bm_valid_test[idx] = list_of_bm_valid_test[idx].detach().cpu().numpy().reshape(list_of_bm_valid_test[idx].shape[0] * list_of_bm_valid_test[idx].shape[1],-1)
                
                    all_bm_valid_test_outputs[idx].extend(list_of_bm_valid_test[idx])
                outputs = torch.stack(dec_out)
                outputs = torch.mean(outputs,axis=0) 
                outputs = outputs[:, :, f_dim:]

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

                pred = outputs.detach().cpu()
                true = batch_x.detach().cpu()

                loss = criterion(pred, true) + ssl_loss_v2(slow_out, batch_x, slow_mark, s1, s2, self.device)
                total_loss.append(loss.item())
                if i == 300: break
        dict_learner_test = {}
        for learner_idx in range(self.args.n_learner):
            dict_learner_test[header_bmnpzreader[learner_idx]] = np.array(all_bm_valid_test_outputs[learner_idx], dtype="object")
        if epoch == 0 :
            # np_input_test_X = test_X[0] # biar punya awalan shape array numpy
            # np_input_test_y = test_y[0]
            # for i in range(1,test_X[0].shape[0]):
            #     np_input_test_X = np.append(np_input_test_X,test_X[i])
            # for j in range(1,test_y[0].shape[0]):
            #     np_input_test_y = np.append(np_input_test_y,test_y[j])
            np_input_test_X = np.array(test_X,dtype="object")
            np_input_test_y = np.array(test_y,dtype="object")
            np.save(f"{self.args.root_path}test_X.npy",np_input_test_X)
            np.save(f"{self.args.root_path}test_y.npy",np_input_test_y)
        total_loss = np.average(total_loss)
        return total_loss,dict_learner_test
    
    def vali(self, vali_data, vali_loader, criterion, header_bmnpzreader, epoch, flag):
        self.model.eval()
        if flag == "valid":
            valid_X = []
            total_loss,dict_learner_valid = self.vali_step_validator(vali_loader, criterion, header_bmnpzreader, valid_X, epoch, flag)
            self.model.train()
            return total_loss,dict_learner_valid
        else:
            test_X,test_y = [], []
            total_loss,dict_learner_test = self.vali_step_tester(vali_loader, criterion, header_bmnpzreader, test_X, test_y, epoch, flag)
            self.model.train()
            return total_loss,dict_learner_test

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)
        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        slow_model_optim = self._select_slow_optimizer()
        criterion = self._select_criterion()
        f = open("training_mantra_anomaly_detection.txt", 'a')
        f_csv = open("training_mantra_anomaly_detection.csv","a")
        csvreader = csv.writer(f_csv)
        
        header_bmnpzreader = [f"learner{i+1}" for i in range(self.args.n_learner)]
        train_X = []
        for epoch in tqdm(list(range(self.args.train_epochs))):
            iter_count = 0
            train_loss = []
            all_bm_train_outputs = [[] for _ in range(self.args.n_learner)]
            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y) in enumerate(train_loader):
                if epoch == 0:
                    train_X.extend(batch_x.detach().cpu().numpy())

                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                dec_out = []
                list_of_bm_train = [[] for _ in range(self.args.n_learner)]
                f_dim = -1 if self.args.features == 'MS' else 0
                for idx in range(self.args.n_learner):
                    if self.model.name not in ["KBJNet"]:
                        outputs = self.model.forward_1learner(batch_x, idx)
                    else:
                        outputs = self.model.forward_1learner(batch_x.permute(0,2,1),idx) 
                    list_of_bm_train[idx].append(outputs)
                    dec_out.append(outputs)

                for idx in range(self.args.n_learner):
                    list_of_bm_train[idx] = torch.stack(list_of_bm_train[idx])
                    list_of_bm_train[idx] = torch.mean(list_of_bm_train[idx],axis=0)
                    list_of_bm_train[idx] = list_of_bm_train[idx][:,:,f_dim:]
                    list_of_bm_train[idx] = list_of_bm_train[idx].detach().cpu().numpy().reshape(list_of_bm_train[idx].shape[0] * list_of_bm_train[idx].shape[1],-1)

                    all_bm_train_outputs[idx].extend(list_of_bm_train[idx])
                outputs = torch.stack(dec_out)
                outputs = torch.mean(outputs,axis=0)
                outputs = outputs[:, :, f_dim:]
                loss = criterion(outputs, batch_x)
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()
                
                # loss.backward()
                # model_optim.step()
                # Slow Learner
                # loss = 0
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
                loss = loss + ssl_loss_v2(slow_out, batch_x, slow_mark, s1, s2, self.device)

                slow_model_optim.zero_grad()    
                loss.backward()
                slow_model_optim.step()
                model_optim.step()
                if i == 300: break
            if epoch == 0:
                # np_input_train_X = train_X[0] # biar punya awalan shape array numpy
                # for i in range(1,train_X[0].shape[0]):
                #     np_input_train_X = np.append(np_input_train_X,train_X[i])
                np_input_train_X = np.array(train_X, dtype="object")
                np.save(f"{self.args.root_path}train_X.npy",np_input_train_X)
                
                f.write(setting + "  \n")    
                header = [[setting],["Epoch","Cost Time", "Steps", "Train Loss", "Vali Loss", "Test Loss"]]
                csvreader.writerows(header)
            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            f.write("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            f.write("\n")
            train_loss = np.average(train_loss)
            vali_loss,dict_learner_valid = self.vali(vali_data, vali_loader, criterion, header_bmnpzreader, epoch, "valid")
            test_loss,dict_learner_test = self.vali(test_data, test_loader, criterion, header_bmnpzreader, epoch, "test")

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
        
        dict_learner = {}
        for learner_idx in range(self.args.n_learner):
            dict_learner[header_bmnpzreader[learner_idx]] = np.array(all_bm_train_outputs[learner_idx], dtype="object")
        np.savez(f"{self.args.root_path}bm_train_preds.npz", **dict_learner)
        np.savez(f"{self.args.root_path}bm_valid_preds.npz", **dict_learner_valid)
        np.savez(f"{self.args.root_path}bm_test_preds.npz", **dict_learner_test)
        # best_model_path = path + '/' + 'checkpoint.pth'
        # self.model.load_state_dict(torch.load(best_model_path)) # load_state_dict
        f.write("\n")
        csvreader.writerow([])
        f.close()
        return
    
    def test(self, setting, test=1):
        _, test_loader = self._get_data(flag='test')
        _, train_loader = self._get_data(flag='train')

        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        attens_energy = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        self.anomaly_criterion = nn.MSELoss(reduce=False)
        # (1) stastic on the TRAIN SET
        with torch.no_grad():
            for i, (batch_x, _) in enumerate(train_loader):
                batch_x = batch_x.float().to(self.device)
                # reconstruction
                if self.model.name not in ["KBJNet"]:
                    outputs = self.model(batch_x)
                else:
                    outputs = self.model(batch_x.permute(0,2,1)) 
                # criterion
                loss = self.anomaly_criterion(batch_x, outputs)
                score = torch.mean(loss, dim=-1)
                score = score.detach().cpu().numpy()
                attens_energy.append(score)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        train_energy = np.array(attens_energy)

        # (2) find the threshold TEST SET
        attens_energy = []
        test_labels = []
        for i, (batch_x, batch_y) in enumerate(test_loader):
            batch_x = batch_x.float().to(self.device)
            # reconstruction
            if self.model.name not in ["KBJNet"]:
                outputs = self.model(batch_x)
            else:
                outputs = self.model(batch_x.permute(0,2,1)) 
            # criterion
            lossT = self.anomaly_criterion(batch_x, outputs)
            score = torch.mean(lossT, dim=-1)
            score = score.detach().cpu().numpy()
            attens_energy.append(score)
            test_labels.append(batch_y)

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

        np.save(folder_path + 'metrics.npy', np.array([accuracy, precision, recall, f_score]))
        np.save(folder_path + 'pred.npy', pred)
        np.save(folder_path + 'groundtruth.npy', gt)
        return
    
    # def test_1learner(self, setting, test=0, idx=0):
    #     test_data, test_loader = self._get_data(flag='test')
    #     train_data, train_loader = self._get_data(flag='train')

    #     if test:
    #         print('loading model')
    #         self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

    #     attens_energy = []
    #     folder_path = './test_results/' + setting + '/'
    #     if not os.path.exists(folder_path):
    #         os.makedirs(folder_path)

    #     self.model.eval()
    #     self.anomaly_criterion = nn.MSELoss(reduce=False)
    #     # (1) stastic on the TRAIN SET
    #     with torch.no_grad():
    #         for i, (batch_x, _) in enumerate(train_loader):
    #             batch_x = batch_x.float().to(self.device)
    #             # encoder - decoder
    #             if self.args.use_multi_gpu and self.args.use_gpu:
    #                 if self.model.name not in ["KBJNet"]:
    #                     outputs = self.model.module.forward_1learner(batch_x,idx=idx)
    #                 else:
    #                     outputs = self.model.module.forward_1learner(batch_x.permute(0,2,1),idx=idx)
    #             else:
    #                 if self.model.name not in ["KBJNet"]:
    #                     outputs = self.model.forward_1learner(batch_x,idx=idx)
    #                 else:
    #                     outputs = self.model.forward_1learner(batch_x.permute(0,2,1),idx=idx)

    #             loss = self.anomaly_criterion(batch_x, outputs)
    #             score = torch.mean(loss, dim=-1)
    #             score = score.detach().cpu().numpy()
    #             attens_energy.append(score)

    #     attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
    #     train_energy = np.array(attens_energy)

    #     # (2) find the threshold TEST SET
    #     attens_energy = []
    #     test_labels = []
    #     for i, (batch_x, batch_y) in enumerate(test_loader):
    #         batch_x = batch_x.float().to(self.device)
    #         # reconstruction
    #         if self.args.use_multi_gpu and self.args.use_gpu:
    #             if self.model.name not in ["KBJNet"]:
    #                 outputs = self.model.module.forward_1learner(batch_x,idx=idx)
    #             else:
    #                 outputs = self.model.module.forward_1learner(batch_x.permute(0,2,1),idx=idx)
    #         else:
    #             if self.model.name not in ["KBJNet"]:
    #                 outputs = self.model.forward_1learner(batch_x,idx=idx)
    #             else:
    #                 outputs = self.model.forward_1learner(batch_x.permute(0,2,1),idx=idx)
    #         # criterion
    #         lossT = self.anomaly_criterion(batch_x, outputs)
    #         score = torch.mean(lossT, dim=-1)
    #         score = score.detach().cpu().numpy()
    #         attens_energy.append(score)
    #         test_labels.append(batch_y)

    #     attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
    #     test_energy = np.array(attens_energy)
    #     combined_energy = np.concatenate([train_energy, test_energy], axis=0)

    #     threshold = np.percentile(combined_energy, 100 - self.args.anomaly_ratio)

    #     print("Threshold :", threshold)

    #     # (3) evaluation on the test set
    #     pred = (test_energy > threshold).astype(int)
    #     test_labels = np.concatenate(test_labels, axis=0).reshape(-1)
    #     test_labels = np.array(test_labels)

    #     gt = test_labels.astype(int)
    #     print("pred:   ", pred.shape)
    #     print("gt:     ", gt.shape)

    #     # (4) detection adjustment
    #     gt, pred = adjustment(gt, pred) #gt == label

    #     pred = np.array(pred)
    #     gt = np.array(gt)
    #     print("pred: ", pred.shape)
    #     print("gt:   ", gt.shape)

    #     accuracy = accuracy_score(gt, pred)
    #     precision, recall, f_score, support = precision_recall_fscore_support(gt, pred, average='binary')
    #     print("Accuracy : {:0.4f}, Precision    : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(
    #         accuracy, precision,
    #         recall, f_score))

    #     # result_anomaly_detection_mantra.txt
    #     f = open("result_anomaly_detection_mantra.txt", 'a')
    #     f.write(setting + "  \n")
    #     f.write("Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(
    #         accuracy, precision,
    #         recall, f_score))
    #     f.write('\n')
    #     f.write('\n')
    #     f.close()
    #     #result_anomaly_detection_mantra.csv
    #     f_csv = open("result_anomaly_detection_mantra.csv","a")
    #     csvreader = csv.writer(f_csv)
    #     datas = [[setting],["Accuracy","Precision","Recall","F-score"],[round(accuracy,4),round(precision,4),round(recall,4),round(f_score,4)]]
    #     csvreader.writerows(datas)
    #     return