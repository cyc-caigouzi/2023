import os
import time
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import DLinear, SCINet, MTSMixer, MTSMatrix, MTSAttn, FNet, Transformer_lite, MTSD, Tide
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric, R2

warnings.filterwarnings('ignore')

non_transformer = ['DLinear', 'SCINet', 'MTSMixer', 'MTSMatrix', 'MTSAttn', 'FNet', 'Transformer_lite', 'MTSD', 'Tide']

class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)

    def _build_model(self):
        model_dict = {
            #'Transformer': Transformer,
            'DLinear': DLinear,
            'SCINet': SCINet,
            'MTSMixer': MTSMixer,
            'MTSMatrix': MTSMatrix,
            'MTSAttn': MTSAttn,
            'FNet': FNet,
            'Transformer_lite': Transformer_lite,
            'MTSD': MTSD,
            'Tide': Tide
        }
        model = model_dict[self.args.model].Model(self.args).float()
        
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)

        total = sum([param.nelement() for param in model.parameters()])
        print('Number of parameters: %.2fM' % (total / 1e6))

        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)

        return data_set, data_loader

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for _, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                #batch_x-seq_x   batch_y-seq_y batch_x_mark-seq_x_mark, batch_y_mark-seq_y_mark
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                outputs = self.model(batch_x, batch_x_mark, batch_y_mark)

                outputs = outputs[:, -self.args.H:]
                outputs = outputs.squeeze(-1)
                batch_y = batch_y[:, -self.args.H:].to(self.device)
                loss = criterion(outputs.detach().cpu(), batch_y.detach().cpu())
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

        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        criterion = nn.MSELoss()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                #print("batch_x.shape(exp):", batch_x.shape)
                #print("batch_y.shape(exp):", batch_y.shape)
                #print("batch_x_mark(exp):", batch_x_mark.shape)
                #print("batch_y_mark(exp):", batch_y_mark.shape)

                outputs = self.model(batch_x, batch_x_mark, batch_y_mark)

                outputs = outputs[:, -self.args.H:]
                batch_y = batch_y[:, -self.args.H:].to(self.device)
                #print("outputs shape:",outputs.shape)
                outputs = outputs.squeeze(-1)
                #print("batch_y shape:", batch_y.shape)
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
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        _, test_loader = self._get_data(flag='test')

        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth'), map_location=self.device))

        preds = []
        trues = []
        inputx = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for idx, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                outputs = self.model(batch_x, batch_x_mark, batch_y_mark)
                #print("outputs shape1:", outputs.shape)
                outputs = torch.squeeze(outputs, dim=2)
                #print("outputs shape2:", outputs.shape)
                outputs = outputs[:, -self.args.H:]
                #print("outputs shape3:", outputs.shape)
                batch_y = batch_y[:, -self.args.H:].to(self.device)
                #print("batch_y shape:", batch_y.shape)

                pred = outputs.detach().cpu().numpy()
                true = batch_y.detach().cpu().numpy()
                preds.append(pred)
                trues.append(true)
                inputx.append(batch_x.detach().cpu().numpy())

                if idx % self.args.seg == 0:
                    input = batch_x.detach().cpu().numpy()
                    gt = np.concatenate((input[0, :], true[0, :]), axis=0)
                    pd = np.concatenate((input[0, :], pred[0, :]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(idx) + '.pdf'))

        print("input shape:", input.shape)
        print("true shape:", true.shape)

        print("input dimensions:", input.ndim)
        print("true dimensions:", true.ndim)
        preds = np.array(preds)
        trues = np.array(trues)
        inputx = np.array(inputx)

        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        inputx = inputx.reshape(-1, inputx.shape[-2], inputx.shape[-1])

        mse, mae, mape, r_squared = metric(preds, trues)
        print('mse:{:.4f}, mae:{:.4f},  mape:{:.4f}, R2:{:.4f}'.format(mse, mae, mape, r_squared))

        return

    def predict(self, setting, load=False):
        _, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []
        self.model.eval()
        with torch.no_grad():
            for _, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                outputs = self.model(batch_x, batch_x_mark, batch_y_mark)

                preds.append(outputs.detach().cpu().numpy())

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)

        return
