from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjustment, visual
from utils.metrics import NegativeCorr
from models.PropPrototype import MultiHeadURT
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
import numpy as np
import torch
import torch.nn as nn
from torch import optim

import os
import time

import warnings
import matplotlib.pyplot as plt
import numpy as np
import random

import copy
# import h5py

warnings.filterwarnings('ignore')


class Opt_URT_Anomaly(Exp_Basic):
    def __init__(self, args):
        super(Opt_URT_Anomaly, self).__init__(args)

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
    def _build_urt(self):
        URT = MultiHeadURT(key_dim=self.args.win_size , query_dim=self.args.win_size*self.args.enc_in, hid_dim=4096, temp=1, att="cosine", n_head=self.args.urt_heads).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            URT = nn.DataParallel(URT, device_ids=self.args.device_ids)
        return URT
    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader
    
    def _select_urt_optimizer(self):
        urt_optim = optim.Adam(self.URT.parameters(), lr=0.0001)
        return urt_optim

    def _select_criterion(self):
        if self.args.loss_type == "negative_corr":
            criterion = NegativeCorr(self.args.correlation_penalty)
        else:
            criterion = nn.MSELoss()
        return criterion

    def vali2(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        self.URT.eval()
        with torch.no_grad():
            for i, (batch_x, _) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                # FAST LEARNER
                dec_out = []
                for idx in range(0,self.args.n_learner):
                    outputs = self.model.forward_1learner(batch_x,idx=idx)
                    dec_out.append(outputs)

                dec_out = torch.stack(dec_out)
                dec_out2 = torch.mean(dec_out,axis=1)
                dec_out2 = dec_out2.reshape(dec_out2.shape[0],dec_out2.shape[2],dec_out2.shape[1])
                #Forward URT
                urt_out = self.URT(dec_out2)

                a,b,c,d = dec_out.shape
                fin_out = torch.zeros([b,c,d]).cuda() if self.args.use_gpu else torch.zeros([b,c,d])
                for k in range(0,d):
                    for l in range(0,a):
                        fin_out[:,:,k] = fin_out[:,:,k] + (dec_out[l,:,:,k] * urt_out[l,k])
                
                # SLOW LEARNER
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

                outputs = fin_out + slow_out

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, :, f_dim:]

                pred = outputs.detach().cpu()
                true = batch_x.detach().cpu()

                loss = criterion(pred, true)
                total_loss.append(loss)

        total_loss = np.average(total_loss)
        self.URT.train()
        return total_loss

    def train_urt(self, setting):
        
        self.model.load_state_dict(torch.load(os.path.join(str(self.args.checkpoints) + setting, 'checkpoint.pth')))
        # self.slow_model.load_state_dict(torch.load(os.path.join(str(self.args.checkpoints) + setting, 'checkpoint_slow_learner.pth')))
        
        print("Train URT layer >>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True, urt=True)

        urt_optim = self._select_urt_optimizer()
        criterion = self._select_criterion()

        f = open("training_urt_anomaly_detection.txt", 'a')
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.eval()
            self.slow_model.eval()
            self.URT.train()
            epoch_time = time.time()

            for i, (batch_x, _) in enumerate(train_loader):
                iter_count += 1
                urt_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)

                # FAST LEARNER
                dec_out = []
                for idx in range(self.args.n_learner):
                    if self.args.use_multi_gpu and self.args.use_gpu:
                        outputs = self.model.module.forward_1learner(batch_x, idx=idx)
                        if self.model.name not in ["KBJNet"]:
                            outputs = self.model.module.forward_1learner(batch_x, idx=idx)
                        else:
                            outputs = self.model.module.forward_1learner(batch_x.permute(0,2,1), idx=idx)
                    else:
                        if self.model.name not in ["KBJNet"]:
                            outputs = self.model.forward_1learner(batch_x, idx=idx)
                        else:
                            outputs = self.model.forward_1learner(batch_x.permute(0,2,1), idx=idx)
                    dec_out.append(outputs)

                dec_out = torch.stack(dec_out)
                dec_out2 = torch.mean(dec_out,axis=1)
                dec_out2 = dec_out2.reshape(dec_out2.shape[0],dec_out2.shape[2],dec_out2.shape[1])
                #Forward URT
                urt_out = self.URT(dec_out2)

                a,b,c,d = dec_out.shape
                fin_out = torch.zeros([b,c,d]).cuda() if self.args.use_gpu else torch.zeros([b,c,d])
                for k in range(0,d):
                    for l in range(0,a):
                        fin_out[:,:,k] = fin_out[:,:,k] + (dec_out[l,:,:,k] * urt_out[l,k])

                # # SLOW LEARNER
                # s0,s1,s2 = batch_x.shape
                # randuniform = torch.empty(s0,s1,s2).uniform_(0, 1)
                # m_ones = torch.ones(s0,s1,s2)
                # slow_mark = torch.bernoulli(randuniform)
                # batch_x_slow = batch_x.clone()
                # batch_x_slow = batch_x_slow * (m_ones-slow_mark).to(self.device)

                # if self.slow_model.name not in ["KBJNet"]:
                #     slow_out = self.slow_model.forward(batch_x_slow)
                # else:
                #     slow_out = self.slow_model.forward(batch_x_slow.permute(0,2,1))

                # outputs = fin_out + slow_out

                outputs = fin_out 
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, :, f_dim:]

                pred = outputs.detach().cpu()
                true = batch_x.detach().cpu()

                loss = criterion(pred, true)
                loss.requires_grad = True
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                urt_optim.step()
            if epoch == 0:
                f.write(setting + "  \n")  
            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            f.write("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            f.write("\n")
            train_loss = np.average(train_loss)
            vali_loss = self.vali2(vali_data, vali_loader, criterion)
            test_loss = self.vali2(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            f.write("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            f.write("\n")
            early_stopping(vali_loss, self.URT, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
        f.close()
        return

    def test2(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        train_data, train_loader = self._get_data(flag='train')

        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))
            # self.slow_model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))
            self.URT.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint_urt.pth')))

        attens_energy = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        self.slow_model.eval()
        self.URT.eval()
        self.anomaly_criterion = nn.MSELoss(reduce=False)

        #(1) stastic on the TRAIN SET
        with torch.no_grad():
            for i, (batch_x, _) in enumerate(train_loader):
                batch_x = batch_x.float().to(self.device)
                # encoder - decoder
                dec_out = []
                for idx in range(self.args.n_learner):
                    if self.args.use_multi_gpu and self.args.use_gpu:
                        if self.model.name not in ["KBJNet"]:
                            outputs = self.model.module.forward_1learner(batch_x, idx=idx)
                        else:
                            outputs = self.model.module.forward_1learner(batch_x.permute(0,2,1), idx=idx)
                    else:
                        if self.model.name not in ["KBJNet"]:
                            outputs = self.model.forward_1learner(batch_x, idx=idx)
                        else:
                            outputs = self.model.forward_1learner(batch_x.permute(0,2,1), idx=idx)
                    dec_out.append(outputs)

                dec_out = torch.stack(dec_out)
                dec_out2 = torch.mean(dec_out,axis=1)
                dec_out2 = dec_out2.reshape(dec_out2.shape[0],dec_out2.shape[2],dec_out2.shape[1])
                # Test URT
                urt_out = self.URT(dec_out2)

                a,b,c,d = dec_out.shape
                fin_out = torch.zeros([b,c,d]).cuda() if self.args.use_gpu else torch.zeros([b,c,d])
                for k in range(0,d):
                    for l in range(0,a):
                        fin_out[:,:,k] = fin_out[:,:,k] + (dec_out[l,:,:,k] * urt_out[l,k])
                # # SLOW LEARNER
                # s0,s1,s2 = batch_x.shape
                # randuniform = torch.empty(s0,s1,s2).uniform_(0, 1)
                # m_ones = torch.ones(s0,s1,s2)
                # slow_mark = torch.bernoulli(randuniform)
                # batch_x_slow = batch_x.clone()
                # batch_x_slow = batch_x_slow * (m_ones-slow_mark).to(self.device)

                # if self.slow_model.name not in ["KBJNet"]:
                #     slow_out = self.slow_model.forward(batch_x_slow)
                # else:
                #     slow_out = self.slow_model.forward(batch_x_slow.permute(0,2,1))
                
                # outputs = fin_out + slow_out
                outputs = fin_out
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, :, f_dim:]
                # Check Anomaly Train Loader
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
            dec_out = []
            for idx in range(0,self.args.n_learner):
                if self.args.use_multi_gpu and self.args.use_gpu:
                    if self.model.name not in ["KBJNet"]:
                        outputs = self.model.module.forward_1learner(batch_x, idx=idx)
                    else:
                        outputs = self.model.module.forward_1learner(batch_x.permute(0,2,1), idx=idx)
                else:
                    if self.model.name not in ["KBJNet"]:
                        outputs = self.model.forward_1learner(batch_x, idx=idx)
                    else:
                        outputs = self.model.forward_1learner(batch_x.permute(0,2,1), idx=idx)
                dec_out.append(outputs)

            dec_out = torch.stack(dec_out)
            dec_out2 = torch.mean(dec_out,axis=1)
            dec_out2 = dec_out2.reshape(dec_out2.shape[0],dec_out2.shape[2],dec_out2.shape[1])
            # Test URT
            urt_out = self.URT(dec_out2)
            a,b,c,d = dec_out.shape
            fin_out = torch.zeros([b,c,d]).cuda() if self.args.use_gpu else torch.zeros([b,c,d])
            for k in range(0,d):
                for l in range(0,a):
                    fin_out[:,:,k] = fin_out[:,:,k] + (dec_out[l,:,:,k] * urt_out[l,k])
            
            # # SLOW LEARNER
            # s0,s1,s2 = batch_x.shape
            # randuniform = torch.empty(s0,s1,s2).uniform_(0, 1)
            # m_ones = torch.ones(s0,s1,s2)
            # slow_mark = torch.bernoulli(randuniform)
            # batch_x_slow = batch_x.clone()
            # batch_x_slow = batch_x_slow * (m_ones-slow_mark).to(self.device)

            # if self.slow_model.name not in ["KBJNet"]:
            #     slow_out = self.slow_model.forward(batch_x_slow)
            # else:
            #     slow_out = self.slow_model.forward(batch_x_slow.permute(0,2,1))
            
            # outputs = fin_out + slow_out
            outputs = fin_out
            f_dim = -1 if self.args.features == 'MS' else 0
            outputs = outputs[:, :, f_dim:]

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
        f = open("result_anomaly_detection_mantra.txt", 'a')
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


        # fname = f"{setting}_dataset.h5"
        # hf = h5py.File(fname, 'w')
        # hf.create_dataset('preds', data=pred)
        # hf.create_dataset('groundtruths', data=gt)
        # hf.close()

        #     np.savetxt(fname, trues[:,:,col],delimiter=",")

        # for col in range (0,preds.shape[-1]):
        # # fname = setting +"_preds_" + str(col) + ".dat"
        #     fname = "ZZZ_Mantra_ETTm2_pl"+str(self.args.pred_len)+"_col"+str(col)+"_preds.csv"
        #     np.savetxt(fname, trues[:,:,col],delimiter=",")
        #     fname = "ZZZ_Mantra_ETTm2_pl"+str(self.args.pred_len)+"_col"+str(col)+"_preds.csv"
        #     np.savetxt(fname, trues[:,:,col],delimiter=",")
        # preds.tofile(fname)
        # # fname = setting +"_trues_" + str(col) + ".dat"
        # fname = "ZZZ_Mantra_ETTm2_pl"+str(self.args.pred_len)+"_trues.dat"
        # trues.tofile(fname)
        # np.savetxt(fname, trues[:,:,col],delimiter=",")

        return

    #Tanpa URT
    # def test_wo_urt(self, setting, test=0):
    #     test_data, test_loader = self._get_data(flag='test')
    #     if test:
    #         print('loading model')
    #         self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

    #     preds = []
    #     trues = []
    #     folder_path = './test_results/' + setting + '/'
    #     if not os.path.exists(folder_path):
    #         os.makedirs(folder_path)

    #     self.model.eval()
    #     isFirst = True
    #     with torch.no_grad():
    #         for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
    #             batch_x = batch_x.float().to(self.device)
    #             batch_y = batch_y.float().to(self.device)

    #             batch_x_mark = batch_x_mark.float().to(self.device)
    #             batch_y_mark = batch_y_mark.float().to(self.device)

    #             # decoder input
    #             dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
    #             dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
    #             # encoder - decoder
    #             if self.args.use_amp:
    #                 with torch.cuda.amp.autocast():
    #                     if self.args.output_attention:
    #                         outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
    #                     else:
    #                         outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
    #             else:
    #                 if self.args.output_attention:
    #                     outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

    #                 else:
    #                     outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

    #             f_dim = -1 if self.args.features == 'MS' else 0
    #             outputs = outputs[:, -self.args.pred_len:, f_dim:]
    #             batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
    #             outputs = outputs.detach().cpu().numpy()
    #             batch_y = batch_y.detach().cpu().numpy()

    #             pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()
    #             true = batch_y  # batch_y.detach().cpu().numpy()  # .squeeze()

    #             if isFirst:
    #                 isFirst = False
    #                 preds = np.array(pred)
    #                 trues = np.array(true)

    #             else:
    #                 preds = np.concatenate((preds,pred), axis=0)
    #                 trues = np.concatenate((trues,true), axis=0)
                
    #             if i % 20 == 0:
    #                 input = batch_x.detach().cpu().numpy()
    #                 gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
    #                 pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
    #                 visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

    #     preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
    #     trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])


    #     # result save
    #     folder_path = './results/' + setting + '/'
    #     if not os.path.exists(folder_path):
    #         os.makedirs(folder_path)

    #     print('test shape:', preds.shape, trues.shape)
    #     mae, mse, rmse, mape, mspe = metric(preds, trues)
    #     print('mse:{}, mae:{}'.format(mse, mae))
    #     f = open("result.txt", 'a')
    #     f.write(setting + "  \n")
    #     f.write('mse:{}, mae:{}'.format(mse, mae))
    #     f.write('\n')
    #     f.write('\n')
    #     f.close()

    #     np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
    #     np.save(folder_path + 'pred.npy', preds)
    #     np.save(folder_path + 'true.npy', trues)

      
    #     return