from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, adjustment, cal_accuracy
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import accuracy_score, roc_auc_score
import torch.multiprocessing
from tqdm import tqdm

torch.multiprocessing.set_sharing_strategy('file_system')
import torch
import torch.nn as nn
from torch import optim
import os
import time
from datetime import datetime
import warnings
import numpy as np

warnings.filterwarnings('ignore')


class Exp_Anomaly_Detection(Exp_Basic):
    def __init__(self, args):
        super(Exp_Anomaly_Detection, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.RAdam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        def criterion(outputs, batch_y):
            pos_sum = torch.sum(batch_y)
            neg_sum = torch.sum(1-batch_y)
            pos_weight = ((1+neg_sum) / (1+pos_sum)).item()
            positive_class_weight = torch.tensor([pos_weight]).to(outputs.device)
            loss_func = nn.BCEWithLogitsLoss(pos_weight=positive_class_weight)
            loss = loss_func(outputs, batch_y.float())
            return loss
        return criterion
    
    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(tqdm(vali_loader)):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                outputs = self.model(batch_x, None, None, None)
                outputs = outputs[:, :, 0]
                pred = outputs.detach()
                true = batch_y.detach()

                loss = criterion(pred, true)
                total_loss.append(loss.item())
                
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        # vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y) in enumerate(tqdm(train_loader)):
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                outputs = self.model(batch_x, None, None, None)
                outputs = outputs[:, :, 0]
                
                loss = criterion(outputs, batch_y)
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
                
            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            # vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Test Loss: {3:.7f}".format(
                epoch + 1, train_steps, train_loss, test_loss))
            early_stopping(test_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            adjust_learning_rate(model_optim, epoch + 1, self.args)
            
        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        # train_data, train_loader = self._get_data(flag='train')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        attens_energy = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()

        preds = []
        trues = []
        for i, (batch_x, batch_y) in enumerate(tqdm(test_loader)):
            batch_x = batch_x.float().to(self.device)
            # reconstruction
            outputs = self.model(batch_x, None, None, None)
            outputs = outputs[:, :, 0]
            preds.append(outputs.detach().cpu())
            trues.append(batch_y.detach().cpu())

        preds = torch.cat(preds, 0).flatten()
        trues = torch.cat(trues, 0).flatten().cpu().numpy()
        preds = torch.sigmoid(preds)
        auc_score = roc_auc_score(trues, preds)
        best_f1, best_precision, best_recall = -1, -1, -1
        for thres in np.linspace(0.1, 0.9, 9):
            preds_int = (preds > thres).cpu().numpy()  # (total_samples,) int class index for each sample
            accuracy = cal_accuracy(preds_int, trues)
            precision = precision_score(trues, preds_int, average='binary')
            recall = recall_score(trues, preds_int, average='binary')
            f1 = f1_score(trues, preds_int, average='binary')
            if f1 > best_f1:
                best_f1 = f1
                best_precision = precision
                best_recall = recall
        
        folder_path = '/home/jovyan/results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            
        report = "Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f}, AUC: {:0.4f} ".format(
            accuracy, best_precision,
            best_recall, best_f1, auc_score)
        print(report)
        
        current_time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        file_name=f'{current_time_str}.txt'
        f = open(os.path.join(folder_path,file_name), 'a')
        f.write(setting + "  \n")
        f.write(report)
        f.write('\n')
        f.write('\n')
        f.close()
        return
