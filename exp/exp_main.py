from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, adjustment, plotter, smooth
from utils.pot import pot_eval
from utils.diagnosis import ndcg, hit_att
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from matplotlib.backends.backend_pdf import PdfPages
import torch.multiprocessing
from tqdm import tqdm
from pprint import pprint
import matplotlib.pyplot as plt

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
        # model = self.model_dict["MaelNet"].Model(self.args).float().to(self.device)

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.AdamW(self.model.parameters(), lr=self.args.learning_rate, weight_decay=1e-5)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, _) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)

                if self.model.name not in ["KBJNet"]:
                    outputs = self.model(batch_x)
                else:
                    outputs = self.model(batch_x.permute(0,2,1)) 

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, :, f_dim:]

                pred = outputs.detach().cpu()
                true = batch_x.detach().cpu()

                loss = criterion(pred, true)
                total_loss.append(loss)
                
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

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        f = open("training_anomaly_detection.txt", 'a')
        f_csv = open("training_anomaly_detection.csv","a")
        csvreader = csv.writer(f_csv)
        for epoch in tqdm(list(range(self.args.train_epochs))):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, _) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                if self.model.name not in ["KBJNet"]:
                    outputs = self.model(batch_x)
                else:
                    outputs = self.model(batch_x.permute(0,2,1)) 
                 
                f_dim = -1 if self.args.features == 'MS' else 0
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

                loss.backward()
                model_optim.step()
            if epoch == 0:
                f.write(setting + "  \n")
                header = [[setting],["Epoch","Cost Time", "Steps", "Train Loss", "Vali Loss", "Test Loss"]]
                csvreader.writerows(header)
            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            f.write("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion) #tambahin plotter nanti

            data_for_csv = [[epoch + 1, time.time() - epoch_time, train_steps, round(train_loss,7), round(vali_loss,7), round(test_loss,7)],[]]
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            f.write("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            csvreader.writerows(data_for_csv)
            # Saving Model
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            adjust_learning_rate(model_optim, epoch + 1, self.args)
        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path)) # load_state_dict
        csvreader.writerow([])
        return self.model

    def test(self, setting, test=1):
        test_data, test_loader = self._get_data(flag='test')
        _, train_loader = self._get_data(flag='train')

        test_dataset = test_data[test]

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
        # score_plots = []
        # y_pred = []
        for i, (batch_x, batch_y) in enumerate(test_loader):
            batch_x = batch_x.float().to(self.device)
            # reconstruction
            if self.model.name not in ["KBJNet"]:
                outputs = self.model(batch_x)
            else:
                outputs = self.model(batch_x.permute(0,2,1)) 
            # criterion
            lossT = self.anomaly_criterion(batch_x, outputs)
            # variables for plotting
            # score_plot = torch.mean(lossT, dim=0).detach().cpu().numpy()
            # y_pred.append(torch.mean(outputs, dim=0).detach().cpu().numpy())
            # score_plots.append(score_plot)

            score = torch.mean(lossT, dim=-1) #anomaly score
            score = score.detach().cpu().numpy()
            attens_energy.append(score)
            test_labels.append(batch_y)
        # Plotting
        
        # test0 = torch.roll(test_data,0,1) if self.model.name == "KBJNet" else test_data
        # plotter(setting, test0,test_energy,score,test_labels)

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

        print(f"Prediction anomaly before adjustment {np.sum(pred)}")
        # (4) detection adjustment
        gt, pred = adjustment(gt, pred) #gt == label
        print(f"Prediction anomaly after adjustment {np.sum(pred)}")

        pred = np.array(pred)
        gt = np.array(gt)
        print("pred: ", pred.shape)
        print("gt:   ", gt.shape)

        accuracy = accuracy_score(gt, pred)
        precision, recall, f_score, support = precision_recall_fscore_support(gt, pred, average='binary')
        print("Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(
            accuracy, precision,
            recall, f_score))
        
        os.makedirs(os.path.join("plots",setting), exist_ok=True)
        with PdfPages(f'plots/{setting}/confusion_matrix.pdf') as pdf:
            # Compute the confusion matrix
            cm = confusion_matrix(gt, pred)
            # Create a figure for the confusion matrix
            fig, ax = plt.subplots(figsize=(8, 6))
            # Display the confusion matrix
            display = ConfusionMatrixDisplay(confusion_matrix=cm)
            # Set the plot title using the axes object
            ax.set_title(f'Confusion Matrix for Anomaly Detection {self.args.data}')
            # Plot the confusion matrix with customizations
            display.plot(ax=ax)
            # Save the current figure to the PDF
            pdf.savefig(fig)
            # Optionally close the figure to free memory
            plt.close(fig)

        with PdfPages(f'plots/{setting}/times_series_plot.pdf') as pdf:
            fig, (ax1,ax2) = plt.subplots(2,1,figsize=(8,6),sharex=True)
            ax1.set_title('Ground Truth')
            ax2.set_title('Prediction')
            ax1.plot(smooth(train_energy), linewidth=0.3, label="Ground Truth")
            ax2.plot(smooth(test_energy), linewidth=0.3, label="Prediction")

            ax3 = ax1.twinx()
            ax4 = ax2.twinx()

            ax3.fill_between(np.arange(gt.shape[0]), gt, color='blue', alpha=0.3, label='True Anomaly')
            ax4.fill_between(np.arange(pred.shape[0]), pred, color='red', alpha=0.3, label='Predicted Anomaly')
            
            ax3.legend(ncol=2, bbox_to_anchor=(0.6, 1.02))
            ax4.legend(bbox_to_anchor=(1, 1.02))

            ax1.set_yticks([])
            ax2.set_yticks([])
            # Save the current figure to the PDF
            pdf.savefig(fig)
            # Optionally close the figure to free memory
            plt.close(fig)
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