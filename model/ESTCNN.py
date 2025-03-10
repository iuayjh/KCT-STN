import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from sklearn.metrics import roc_auc_score, precision_score, recall_score, accuracy_score
from torch.utils.data import Dataset, DataLoader, random_split
import os
import torch.optim as optim
import dataReader


class ESTCNN(nn.Module):
    def __init__(self):
        super(ESTCNN, self).__init__()

        # core1
        self.conv11 = nn.Sequential(
            nn.Conv2d(1, 16, (1, 3), padding=0),
            nn.BatchNorm2d(16, False),
            nn.ReLU(),
            # nn.Dropout(p=0.25)
        )
        self.conv12 = nn.Sequential(
            nn.Conv2d(16, 16, (1, 3), padding=0),
            nn.BatchNorm2d(16, False),
            nn.ReLU(),
            # nn.Dropout(p=0.25)
        )
        self.conv13 = nn.Sequential(
            nn.Conv2d(16, 16, (1, 3), padding=0),
            nn.BatchNorm2d(16, False),
            nn.ReLU(),
            nn.Dropout(p=0.25)
        )
        self.pooling1 = nn.MaxPool2d((1, 2))

        # core2
        self.conv21 = nn.Sequential(
            nn.Conv2d(16, 32, (1, 3), padding=0),
            nn.BatchNorm2d(32, False),
            nn.ReLU(),
            # nn.Dropout(p=0.25)
        )
        self.conv22 = nn.Sequential(
            nn.Conv2d(32, 32, (1, 3), padding=0),
            nn.BatchNorm2d(32, False),
            nn.ReLU(),
            # nn.Dropout(p=0.25)
        )
        self.conv23 = nn.Sequential(
            nn.Conv2d(32, 32, (1, 3), padding=0),
            nn.BatchNorm2d(32, False),
            nn.ReLU(),
            nn.Dropout(p=0.25)
        )
        self.pooling2 = nn.MaxPool2d((1, 2))

        # core3
        self.conv31 = nn.Sequential(
            nn.Conv2d(32, 64, (1, 3), padding=0),
            nn.BatchNorm2d(64, False),
            nn.ReLU(),
            # nn.Dropout(p=0.25)
        )
        self.conv32 = nn.Sequential(
            nn.Conv2d(64, 64, (1, 3), padding=0),
            nn.BatchNorm2d(64, False),
            nn.ReLU(),
            # nn.Dropout(p=0.25)
        )
        self.conv33 = nn.Sequential(
            nn.Conv2d(64, 64, (1, 3), padding=0),
            nn.BatchNorm2d(64, False),
            nn.ReLU(),
            nn.Dropout(p=0.25)
        )
        self.pooling3 = nn.MaxPool2d((1, 7))

        # fc
        self.flatten1 = nn.Flatten()
        # self.fc1 = nn.Linear(64 * 30 * 5, 50)
        self.fc1 = nn.Linear(64 * 30 * 16, 100)
        self.fc2 = nn.Linear(100, 2)

    def forward(self, x):
        x = x.unsqueeze(1)
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
        # print(x.shape)
        # fc
        x = self.flatten1(x)
        x = F.relu(self.fc1(x))
        # x = F.softmax(self.fc2(x), dim=0)
        x = self.fc2(x)

        return x





# net = ESTCNN().cuda(3)
# print(net(Variable(torch.Tensor(np.random.rand(32, 1, 30, 200)).cuda(3))))

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
        datas = self.data_list[index][0].reshape(1, self.C, self.T).astype('float32')
        labels = self.data_list[index][1]
        return datas, labels

    def __len__(self):
        return len(self.data_list)


if __name__ == '__main__':
    datasource = './data'
    file_list = sorted(os.listdir(datasource))
    cuda_device = 0
    C = 30
    T = 500

    sample_list = []
    for i in range(35, 39):
        print(file_list[i])
        sample_list.extend(dataReader.getEEG(os.path.join(datasource, file_list[i]), '/' + file_list[i]))
    sample_list = dataReader.balance(sample_list)
    tmp_list = dataReader.EEGslice(sample_list)

    # 划分训练、验证、测试数据集的数量
    a = len(tmp_list)
    l = [0.6, 0.2, 0.2]
    for i in range(len(l)):
        l[i] = int(l[i] * a)
    l[2] = a - l[0] - l[1]

    # 统计标签数量
    alert, fatigue = 0, 0
    for it in tmp_list:
        if it[1] == 0:
            fatigue += 1
        else:
            alert += 1
    print(alert, fatigue)

    # 划分训练、验证、测试数据集
    data = GetDataset(tmp_list, C, T)
    train, valid, test = random_split(dataset=data, lengths=l, generator=torch.Generator().manual_seed(0))
    train = DataLoader(train, batch_size=32, shuffle=True, drop_last=True)
    valid = DataLoader(valid, batch_size=32, shuffle=True, drop_last=True)
    test = DataLoader(test, batch_size=32, shuffle=True, drop_last=True)

    # 训练参数
    net = ESTCNN().cuda(cuda_device)
    loss = nn.CrossEntropyLoss()
    lr = 0.0001
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
