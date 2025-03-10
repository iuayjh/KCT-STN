import random

import numpy as np
from sklearn.metrics import roc_auc_score, precision_score, recall_score, accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
import dataReader
import os
from torch.utils.data import Dataset, DataLoader, random_split
from model.KANLinear import KANLinear


class CNNaLSTM(nn.Module):
    def __init__(self):
        super(CNNaLSTM, self).__init__()

        # core1
        self.conv11 = nn.Sequential(
            nn.Conv1d(30, 30, 3, padding=1),
            nn.BatchNorm1d(30, False),
            nn.ReLU(),
            # nn.Dropout(p=0.25)
        )
        self.conv12 = nn.Sequential(
            nn.Conv1d(30, 30, 3, padding=1),
            nn.BatchNorm1d(30, False),
            nn.ReLU(),
            # nn.Dropout(p=0.25)
        )
        self.conv13 = nn.Sequential(
            nn.Conv1d(30, 30, 3, padding=1),
            nn.BatchNorm1d(30, False),
            nn.ReLU(),
            nn.Dropout(p=0.25)
        )
        self.pooling1 = nn.MaxPool1d(2, 2)

        # core2
        self.conv21 = nn.Sequential(
            nn.Conv1d(30, 64, 3, padding=1),
            nn.BatchNorm1d(64, False),
            nn.ReLU(),
            # nn.Dropout(p=0.25)
        )
        self.conv22 = nn.Sequential(
            nn.Conv1d(64, 64, 3, padding=1),
            nn.BatchNorm1d(64, False),
            nn.ReLU(),
            # nn.Dropout(p=0.25)
        )
        self.conv23 = nn.Sequential(
            nn.Conv1d(64, 64, 3, padding=1),
            nn.BatchNorm1d(64, False),
            nn.ReLU(),
            nn.Dropout(p=0.25)
        )
        self.pooling2 = nn.MaxPool1d(2, 2)

        # core3
        self.conv31 = nn.Sequential(
            nn.Conv1d(64, 128, 3, padding=1),
            nn.BatchNorm1d(128, False),
            nn.ReLU(),
            # nn.Dropout(p=0.25)
        )
        self.conv32 = nn.Sequential(
            nn.Conv1d(128, 128, 3, padding=1),
            nn.BatchNorm1d(128, False),
            nn.ReLU(),
            # nn.Dropout(p=0.25)
        )
        self.conv33 = nn.Sequential(
            nn.Conv1d(128, 128, 3, padding=1),
            nn.BatchNorm1d(128, False),
            nn.ReLU(),
            nn.Dropout(p=0.25)
        )
        self.pooling3 = nn.MaxPool1d(2, 2)

        self.lstm = nn.LSTM(25, 50, batch_first=True, num_layers=3)

        # # fc
        self.flatten1 = nn.Flatten()
        self.fc1 = nn.Linear(150, 50)
        self.fc2 = nn.Linear(50, 2)

    def forward(self, x):
        # core1
        x = self.conv11(x)
        x = self.conv12(x)
        x = self.conv13(x)
        x = self.pooling1(x)
        # core2
        x = self.conv21(x)
        x = self.conv22(x)
        x = self.conv23(x)
        x = self.pooling2(x)

        # core3
        x = self.conv31(x)
        x = self.conv32(x)
        x = self.conv33(x)
        x = self.pooling3(x)

        # lstm
        _, (h_n, _) = self.lstm(x)
        x = h_n
        x = x.permute(1, 0, 2)

        # fc
        x = self.flatten1(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x


class CNNaLSTM2(nn.Module):
    def __init__(self):
        super(CNNaLSTM2, self).__init__()

        # core1
        self.conv11 = nn.Sequential(
            nn.Conv1d(30, 30, 3, padding=1),
            nn.BatchNorm1d(30, False),
            nn.ReLU(),
            # nn.Dropout(p=0.25)
        )
        self.conv12 = nn.Sequential(
            nn.Conv1d(30, 30, 3, padding=1),
            nn.BatchNorm1d(30, False),
            nn.ReLU(),
            # nn.Dropout(p=0.25)
        )
        self.conv13 = nn.Sequential(
            nn.Conv1d(30, 30, 3, padding=1),
            nn.BatchNorm1d(30, False),
            nn.ReLU(),
            nn.Dropout(p=0.25)
        )
        self.pooling1 = nn.MaxPool1d(2, 2)

        # core2
        self.conv21 = nn.Sequential(
            nn.Conv1d(30, 64, 3, padding=1),
            nn.BatchNorm1d(64, False),
            nn.ReLU(),
            # nn.Dropout(p=0.25)
        )
        self.conv22 = nn.Sequential(
            nn.Conv1d(64, 64, 3, padding=1),
            nn.BatchNorm1d(64, False),
            nn.ReLU(),
            # nn.Dropout(p=0.25)
        )
        self.conv23 = nn.Sequential(
            nn.Conv1d(64, 64, 3, padding=1),
            nn.BatchNorm1d(64, False),
            nn.ReLU(),
            nn.Dropout(p=0.25)
        )
        self.pooling2 = nn.MaxPool1d(2, 2)

        # core3
        self.conv31 = nn.Sequential(
            nn.Conv1d(64, 128, 3, padding=1),
            nn.BatchNorm1d(128, False),
            nn.ReLU(),
            # nn.Dropout(p=0.25)
        )
        self.conv32 = nn.Sequential(
            nn.Conv1d(128, 128, 3, padding=1),
            nn.BatchNorm1d(128, False),
            nn.ReLU(),
            # nn.Dropout(p=0.25)
        )
        self.conv33 = nn.Sequential(
            nn.Conv1d(128, 128, 3, padding=1),
            nn.BatchNorm1d(128, False),
            nn.ReLU(),
            nn.Dropout(p=0.25)
        )
        self.pooling3 = nn.MaxPool1d(2, 2)

        # self.lstm1 = nn.LSTM(25, 50, batch_first=True)
        # self.lstm2 = nn.LSTM(50, 50, batch_first=True)
        # self.lstm3 = nn.LSTM(50, 50, batch_first=True, num_layers=3)

        self.lstm1 = nn.LSTM(62, 124, batch_first=True)
        self.lstm2 = nn.LSTM(124, 124, batch_first=True)
        self.lstm3 = nn.LSTM(124, 124, batch_first=True, num_layers=3)

        # # fc
        self.flatten1 = nn.Flatten()
        # self.fc1 = nn.Linear(150, 50)
        self.fc1 = nn.Linear(3*124, 50)
        self.fc2 = nn.Linear(50, 2)

    def forward(self, x):
        # core1
        x = self.conv11(x)
        x = self.conv12(x)
        x = self.conv13(x)
        x = self.pooling1(x)
        # core2
        x = self.conv21(x)
        x = self.conv22(x)
        x = self.conv23(x)
        x = self.pooling2(x)

        # core3
        x = self.conv31(x)
        x = self.conv32(x)
        x = self.conv33(x)
        x = self.pooling3(x)

        # lstm
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        _, (h_n, _) = self.lstm3(x)
        x = h_n
        x = x.permute(1, 0, 2)

        # print(x.shape)
        # fc
        x = self.flatten1(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x


class CNNaLSTM3(nn.Module):
    def __init__(self, is_kan=False):
        super(CNNaLSTM3, self).__init__()

        # core1
        self.conv11 = nn.Sequential(
            nn.Conv1d(30, 30, 3, padding=1),
            nn.BatchNorm1d(30, False),
            nn.ReLU(),
            # nn.Dropout(p=0.25)
        )
        self.conv12 = nn.Sequential(
            nn.Conv1d(30, 30, 3, padding=1),
            nn.BatchNorm1d(30, False),
            nn.ReLU(),
            # nn.Dropout(p=0.25)
        )
        self.conv13 = nn.Sequential(
            nn.Conv1d(30, 30, 3, padding=1),
            nn.BatchNorm1d(30, False),
            nn.ReLU(),
            nn.Dropout(p=0.25)
        )
        self.pooling1 = nn.MaxPool1d(2, 2)

        # core2
        self.conv21 = nn.Sequential(
            nn.Conv1d(30, 64, 3, padding=1),
            nn.BatchNorm1d(64, False),
            nn.ReLU(),
            # nn.Dropout(p=0.25)
        )
        self.conv22 = nn.Sequential(
            nn.Conv1d(64, 64, 3, padding=1),
            nn.BatchNorm1d(64, False),
            nn.ReLU(),
            # nn.Dropout(p=0.25)
        )
        self.conv23 = nn.Sequential(
            nn.Conv1d(64, 64, 3, padding=1),
            nn.BatchNorm1d(64, False),
            nn.ReLU(),
            nn.Dropout(p=0.25)
        )
        self.pooling2 = nn.MaxPool1d(2, 2)

        # core3
        self.conv31 = nn.Sequential(
            nn.Conv1d(64, 128, 3, padding=1),
            nn.BatchNorm1d(128, False),
            nn.ReLU(),
            # nn.Dropout(p=0.25)
        )
        self.conv32 = nn.Sequential(
            nn.Conv1d(128, 128, 3, padding=1),
            nn.BatchNorm1d(128, False),
            nn.ReLU(),
            # nn.Dropout(p=0.25)
        )
        self.conv33 = nn.Sequential(
            nn.Conv1d(128, 128, 3, padding=1),
            nn.BatchNorm1d(128, False),
            nn.ReLU(),
            nn.Dropout(p=0.25)
        )
        self.pooling3 = nn.MaxPool1d(2, 2)

        self.lstm1 = nn.LSTM(32, 64, batch_first=True)
        # self.lstm2 = nn.LSTM(64, 128, batch_first=True)
        self.lstm3 = nn.LSTM(64, 64, batch_first=True, num_layers=3)

        # # fc
        self.flatten1 = nn.Flatten()
        # self.fc1 = nn.Linear(150, 50)
        if is_kan:
            self.fc1 = nn.Sequential(KANLinear(
                in_features=3*64,
                out_features=2,
                grid_size=5,
                spline_order=3,
                scale_noise=0.1,
                scale_base=1.0,
                scale_spline=1.0,
                base_activation=torch.nn.SiLU,
                grid_eps=0.02,
                grid_range=[-1, 1]))
        else:
            self.fc1 = nn.Linear(3*64, 2)


    def forward(self, x):
        # print(f'enter-{x.shape}')
        # core1
        x = self.conv11(x)
        x = self.conv12(x)
        x = self.conv13(x)
        x = self.pooling1(x)
        # print(f'core1-{x.shape}')
        # core2
        x = self.conv21(x)
        x = self.conv22(x)
        x = self.conv23(x)
        x = self.pooling2(x)
        # print(f'core2-{x.shape}')
        # core3
        x = self.conv31(x)
        x = self.conv32(x)
        x = self.conv33(x)
        x = self.pooling3(x)
        # print(f'core3-{x.shape}')
        # lstm
        x, _ = self.lstm1(x)
        # print(f'lstm-{x.shape}')
        # x, _ = self.lstm2(x)
        _, (h_n, _) = self.lstm3(x)
        x = h_n
        # print(f'lstm-{x.shape}')
        x = x.permute(1, 0, 2)

        # print(x.shape)
        # fc
        x = self.flatten1(x)
        x = F.relu(self.fc1(x))

        return x

cuda_device = 0


def evaluate(model, datas, params=["acc"]):
    results = []

    predicted = []
    Y = []
    for _, data in enumerate(datas):
        inputs, labels = data
        inputs = inputs.cuda(cuda_device)
        pred = model(inputs)
        pred = torch.max(pred, dim=1)

        predicted.extend(pred[1].data.cpu().numpy())
        Y.extend(labels.data.cpu().numpy())

    for param in params:
        if param == 'acc':
            results.append(accuracy_score(Y, predicted))
        if param == "auc":
            results.append(roc_auc_score(Y, predicted))
        if param == "recall":
            results.append(recall_score(Y, predicted))
        if param == "precision":
            results.append(precision_score(Y, predicted))
        if param == "fmeasure":
            precision = precision_score(Y, predicted)
            recall = recall_score(Y, predicted)
            results.append(2 * precision * recall / (precision + recall))
    return results


class GetDataset(torch.utils.data.Dataset):
    def __init__(self, data_list, C, T):
        super(GetDataset, self).__init__()
        self.data_list = data_list
        self.C = C
        self.T = T

    def __getitem__(self, index):
        datas = self.data_list[index][0].reshape(self.C, self.T).astype('float32')
        labels = self.data_list[index][1]
        return datas, labels

    def __len__(self):
        return len(self.data_list)


def func1():
    datasource = './data'
    file_list = sorted(os.listdir(datasource))
    C = 30
    T = 200

    sample_list = []
    for i in range(61, 62):
        # print(os.path.join(datasource, file_list[i]) + '/' + file_list[i])
        sample_list.extend(dataReader.getEEG(os.path.join(datasource, file_list[i]) + '/' + file_list[i]))
    sample_list = dataReader.balance(sample_list)
    tmp_list = dataReader.EEGslice(sample_list)

    # 划分训练、验证、测试数据集
    a = len(tmp_list)
    l = [0.6, 0.2, 0.2]
    for i in range(len(l)):
        l[i] = int(l[i] * a)
    l[2] = a - l[0] - l[1]

    alert, fatigue = 0, 0
    for it in tmp_list:
        if it[1] == 0:
            fatigue += 1
        else:
            alert += 1
    print(alert, fatigue)

    data = GetDataset(tmp_list, C, T)

    train, valid, test = random_split(dataset=data, lengths=l, generator=torch.Generator().manual_seed(0))
    train = DataLoader(train, batch_size=32, shuffle=True, drop_last=True)
    valid = DataLoader(valid, batch_size=32, shuffle=True, drop_last=True)
    test = DataLoader(test, batch_size=32, shuffle=True, drop_last=True)
    # print(data[0][0].shape)

    net = CNNaLSTM2()
    # p_info, p_list = dataReader.see_info(net)
    # print((p_info, p_list))

    net = net.to(cuda_device)
    loss = nn.CrossEntropyLoss()
    lr = 0.001
    optimizer = optim.Adam(net.parameters(), lr)

    max = 0
    for epoch in range(20):
        print("\nEpoch ", epoch)
        net.train()
        running_loss = 0.0
        for _, data in enumerate(train):
            optimizer.zero_grad()

            inputs, labels = data
            inputs, labels = inputs.to(cuda_device), labels.to(cuda_device)

            outputs = net(inputs)
            l = loss(outputs, labels)
            l.backward()

            optimizer.step()
            running_loss += l.item() / len(train)
        params = ["acc", "auc", "fmeasure"]
        # params = ["acc"]
        print(params)
        print("Training Loss ", running_loss)
        print("Train - ", evaluate(net, train, params))
        print("Validation - ", evaluate(net, valid, params))
        result = evaluate(net, test, params)
        if result[0] > max:
            max = result[0]
        print("Test - ", result)
    print(max)


def func2():
    data = torch.Tensor(np.random.rand(32, 30, 256))
    model = CNNaLSTM3()

    print(model)

    out = model(data)


if __name__ == '__main__':
    func2()
