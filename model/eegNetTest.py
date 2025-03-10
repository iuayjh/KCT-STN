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

cuda_device = 3


class EEGNet(nn.Module):
    '''
        有做实际输入有做转置
        input   (1,1,30,200)
        conv1   (1,1,16,200)
        conv2    (1,4,16,200)
        pooling2    (1,4,4,50)
        conv3   (1,4,4,50)
        pooling3    (1,4,2,12)
        fc  (4 * 2 * int(self.T/16), 1)
    '''
    def __init__(self, C, T):
        super(EEGNet, self).__init__()
        self.T = T
        self.C = C

        # Layer 1
        self.conv1 = nn.Conv2d(1, 16, (1, self.C), padding=0)
        self.batchnorm1 = nn.BatchNorm2d(16, False)

        # Layer 2
        self.padding1 = nn.ZeroPad2d((16, 17, 0, 1))  # 这里右边填充17会导致卷积后矩阵长度增加2，但是由于后面进行步长为4的池化操作
        # 实际维度保持导致维度正常，填15能够保持维度不变
        self.conv2 = nn.Conv2d(1, 4, (2, 32))
        self.batchnorm2 = nn.BatchNorm2d(4, False)
        self.pooling2 = nn.MaxPool2d(2, 4)

        # Layer 3
        self.padding2 = nn.ZeroPad2d((2, 1, 4, 3))
        self.conv3 = nn.Conv2d(4, 4, (8, 4))
        self.batchnorm3 = nn.BatchNorm2d(4, False)
        self.pooling3 = nn.MaxPool2d((2, 4))

        # FC Layer
        # NOTE: This dimension will depend on the number of timestamps per sample in your data.
        # I have 120 timepoints.
        self.fc1 = nn.Linear(4 * 2 * int(self.T/16), 1)

    def forward(self, x):
        x = x.permute(0, 1, 3, 2)
        # Layer 1
        x = F.elu(self.conv1(x))  # 1 16 200 1
        x = self.batchnorm1(x)
        x = F.dropout(x, 0.25)
        # permute可以对任意张量进行转置 例如下 该例为一维张量的例子
        # >> > x = torch.randn(2, 3, 5)
        # >> > x.size()
        # torch.Size([2, 3, 5])
        # >> > x.permute(2, 0, 1).size()
        # torch.Size([5, 2, 3])
        x = x.permute(0, 3, 1, 2)  # 1 1 16 200

        # Layer 2
        x = self.padding1(x)
        x = F.elu(self.conv2(x))  # 1 4 16 200
        x = self.batchnorm2(x)
        x = F.dropout(x, 0.25)
        x = self.pooling2(x)  # 1 4 4 50

        # Layer 3
        x = self.padding2(x)
        x = F.elu(self.conv3(x))  # 1 4 4 50
        x = self.batchnorm3(x)
        x = F.dropout(x, 0.25)
        x = self.pooling3(x)  # 1 4 2 12

        # FC Layer
        x = x.contiguous().view(-1, 4 * 2 * 12)
        x = F.sigmoid(self.fc1(x))
        return x


def evaluate2(model, datas, params=["acc"]):
    results = []

    predicted = []
    Y = []
    for _, data in enumerate(datas):

        inputs, labels = data
        inputs = inputs.cuda(cuda_device)
        pred = model(inputs)

        predicted.extend(pred.data.cpu().numpy())
        Y.extend(labels.data.cpu().numpy())


    for param in params:
        if param == 'acc':
            results.append(accuracy_score(Y, np.round(predicted)))
        if param == "auc":
            results.append(roc_auc_score(Y, predicted))
        if param == "recall":
            results.append(recall_score(Y, np.round(predicted)))
        if param == "precision":
            results.append(precision_score(Y, np.round(predicted)))
        if param == "fmeasure":
            precision = precision_score(Y, np.round(predicted))
            recall = recall_score(Y, np.round(predicted))
            results.append(2 * precision * recall / (precision + recall))
    return results

# 重构的dataset类
class GetDataset(torch.utils.data.Dataset):
    def __init__(self, data_list, C, T):
        super(GetDataset, self).__init__()
        self.data_list = data_list
        self.C = C
        self.T = T

    def __getitem__(self, index):
        datas = self.data_list[index][0].reshape(1, self.C, self.T).astype('float32')
        labels = self.data_list[index][1]
        return datas, labels

    def __len__(self):
        return len(self.data_list)



def run():
    datasource = './data'
    file_list = sorted(os.listdir(datasource))
    C = 30
    T = 200

    sample_list = []
    for i in range(23, 27):
        print(file_list[i])
        sample_list.extend(dataReader.getEEG(os.path.join(datasource, file_list[i]), '/' + file_list[i]))
    sample_list = dataReader.balance(sample_list)
    tmp_list = dataReader.EEGslice(sample_list)

    alert, fatigue = 0, 0
    for it in tmp_list:
        if it[1] == 0:
            fatigue += 1
        else:
            alert += 1
    print(alert, fatigue)

    data = GetDataset(tmp_list, C, T)

    # 划分训练、验证、测试数据集
    a = len(tmp_list)
    l = [0.6, 0.2, 0.2]
    for i in range(len(l)):
        l[i] = int(l[i] * a)
    l[2] = a - l[0] - l[1]

    train, valid, test = random_split(dataset=data, lengths=l, generator=torch.Generator().manual_seed(0))
    train = DataLoader(train, batch_size=32, shuffle=True, drop_last=True)
    valid = DataLoader(valid, batch_size=32, shuffle=True, drop_last=True)
    test = DataLoader(test, batch_size=32, shuffle=True, drop_last=True)

    net = EEGNet(C, T).cuda(cuda_device)
    criterion = nn.BCELoss()
    lr = 0.001
    optimizer = optim.Adam(net.parameters(), lr)

    max = 0
    for epoch in range(20):
        print("\nEpoch ", epoch)
        running_loss = 0.0
        for _, data in enumerate(train):
            inputs, labels = data
            inputs, labels = inputs.cuda(cuda_device), labels.cuda(cuda_device)
            optimizer.zero_grad()
            outputs = net(inputs)
            outputs = outputs.squeeze(-1)
            loss = criterion(outputs, labels.to(torch.float))
            loss.backward()
            optimizer.step()
            running_loss += loss.item() / len(train)
        params = ["acc", "auc", "fmeasure"]
        print(params)
        print("Training Loss ", running_loss)
        print("Train - ", evaluate2(net, train, params))
        print("Validation - ", evaluate2(net, valid, params))
        result = evaluate2(net, test, params)
        if result[0] > max:
            max = result[0]
        print("Test - ", result)
    print(max)
