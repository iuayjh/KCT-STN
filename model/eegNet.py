import numpy as np
from sklearn.metrics import roc_auc_score, precision_score, recall_score, accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim


# class EEGNet(nn.Module):
#
#     # input size = (1,1,120,64)
#     # input size = (1,1,T,C)
#     def __init__(self):
#         super(EEGNet, self).__init__()
#         # self.T = 120
#
#         # Layer 1
#         self.conv1 = nn.Conv2d(1, 16, (1, 64), padding=0)
#         self.batchnorm1 = nn.BatchNorm2d(16, False)
#
#         # Layer 2
#         self.padding1 = nn.ZeroPad2d((16, 17, 0, 1))  # 这里右边填充17会导致卷积后矩阵长度增加2，但是由于后面进行步长为4的池化操作
#         # 实际维度保持导致维度正常，填15能够保持维度不变
#         self.conv2 = nn.Conv2d(1, 4, (2, 32))
#         self.batchnorm2 = nn.BatchNorm2d(4, False)
#         self.pooling2 = nn.MaxPool2d(2, 4)
#
#         # Layer 3
#         self.padding2 = nn.ZeroPad2d((2, 1, 4, 3))
#         self.conv3 = nn.Conv2d(4, 4, (8, 4))
#         self.batchnorm3 = nn.BatchNorm2d(4, False)
#         self.pooling3 = nn.MaxPool2d((2, 4))
#
#         # FC Layer
#         # NOTE: This dimension will depend on the number of timestamps per sample in your data.
#         # I have 120 timepoints.
#         self.fc1 = nn.Linear(4 * 2 * 7, 1)
#
#     def forward(self, x):
#         print('input', x.shape)
#         # Layer 1
#         x = F.elu(self.conv1(x))  # 1 16 120 1
#         x = self.batchnorm1(x)
#         x = F.dropout(x, 0.25)
#
#         x = x.permute(0, 3, 1, 2)  # 1 1 16 120
#         print('permute', x.shape)
#
#         # Layer 2
#         x = self.padding1(x)  # 1 1 17 153
#         x = F.elu(self.conv2(x))  # 1 4 16 122
#         x = self.batchnorm2(x)
#         x = F.dropout(x, 0.25)
#         x = self.pooling2(x)  # 1 4 4 31
#         print('l2', x.shape)
#
#         # Layer 3
#         x = self.padding2(x)    # 1 4 11 34
#         x = F.elu(self.conv3(x))    # 1 4 4 31
#         x = self.batchnorm3(x)
#         x = F.dropout(x, 0.25)
#         x = self.pooling3(x)    # 1 4 2 7
#         print('l3', x.shape)
#
#         # FC Layer
#         x = x.contiguous().view(-1, 4 * 2 * 7)
#         x = F.sigmoid(self.fc1(x))
#         return x

class EEGNet(nn.Module):

    # input size = (1,1,200,30)
    # input size = (1,1,T,C)
    def __init__(self):
        super(EEGNet, self).__init__()
        # self.T = 120

        # Layer 1
        self.conv1 = nn.Conv2d(1, 16, (1, 30), padding=0)
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
        # self.fc1 = nn.Linear(4 * 2 * 12, 2)
        self.fc1 = nn.Linear(4 * 2 * 31, 2)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = x.permute(0, 1, 3, 2)
        # print('input', x.shape)
        # Layer 1
        x = F.elu(self.conv1(x))  # 1 16 120 1
        x = self.batchnorm1(x)
        x = F.dropout(x, 0.25)
        # print('l1', x.shape)

        x = x.permute(0, 3, 1, 2)  # 1 1 16 120
        # print('permute', x.shape)

        # Layer 2
        x = self.padding1(x)  # 1 1 17 153
        x = F.elu(self.conv2(x))  # 1 4 16 122
        x = self.batchnorm2(x)
        x = F.dropout(x, 0.25)
        x = self.pooling2(x)  # 1 4 4 31
        # print('l2', x.shape)

        # Layer 3
        x = self.padding2(x)    # 1 4 11 34
        x = F.elu(self.conv3(x))    # 1 4 4 31
        x = self.batchnorm3(x)
        x = F.dropout(x, 0.25)
        x = self.pooling3(x)    # 1 4 2 7
        # print('l3', x.shape)

        # FC Layer
        # x = x.contiguous().view(-1, 4 * 2 * 12)
        x = x.contiguous().view(-1, 4 * 2 * 31)
        x = F.elu(self.fc1(x))
        return x


# net = EEGNet().cuda(0)
# # print(net.forward(Variable(torch.Tensor(np.random.rand(1, 1, 120, 64)).cuda(0))))
# criterion = nn.BCELoss()
# optimizer = optim.Adam(net.parameters(), 0.01)


def evaluate(model, X, Y, params=["acc"]):
    results = []
    batch_size = 100

    predicted = []

    for i in range(int(len(X) / batch_size)):
        s = i * batch_size
        e = i * batch_size + batch_size

        inputs = Variable(torch.from_numpy(X[s:e]).cuda(0))
        pred = model(inputs)

        predicted.append(pred.data.cpu().numpy())

    inputs = Variable(torch.from_numpy(X).cuda(0))
    predicted = model(inputs)

    predicted = predicted.data.cpu().numpy()


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


# X_train = np.random.rand(100, 1, 200, 30).astype('float32')  # np.random.rand generates between [0, 1)
# y_train = np.round(np.random.rand(100).astype('float32'))  # binary data, so we round it to 0 or 1.
#
# X_val = np.random.rand(100, 1, 200, 30).astype('float32')
# y_val = np.round(np.random.rand(100).astype('float32'))
#
# X_test = np.random.rand(100, 1, 200, 30).astype('float32')
# y_test = np.round(np.random.rand(100).astype('float32'))
#
# batch_size = 32
#
# for epoch in range(1):  # loop over the dataset multiple times
#     print("\nEpoch ", epoch)
#
#     running_loss = 0.0
#     for i in range(int(len(X_train) / batch_size) - 1):
#         s = i * batch_size
#         e = i * batch_size + batch_size
#
#         inputs = torch.from_numpy(X_train[s:e])
#         labels = torch.FloatTensor(np.array([y_train[s:e]]).T * 1.0)
#
#         # wrap them in Variable
#         inputs, labels = Variable(inputs.cuda(0)), Variable(labels.cuda(0))
#
#         # zero the parameter gradients
#         optimizer.zero_grad()
#
#         # forward + backward + optimize
#         outputs = net(inputs)
#         loss = criterion(outputs, labels)
#         loss.backward()
#
#         optimizer.step()
#
#         running_loss += loss.item()
#
#     # Validation accuracy
#     params = ["acc", "auc", "fmeasure"]
#     print(params)
#     print("Training Loss ", running_loss)
#     print("Train - ", evaluate(net, X_train, y_train, params))
#     print("Validation - ", evaluate(net, X_val, y_val, params))
#     print("Test - ", evaluate(net, X_test, y_test, params))



