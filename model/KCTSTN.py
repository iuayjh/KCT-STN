import random

import numpy as np
from sklearn.metrics import roc_auc_score, precision_score, recall_score, accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import copy
import torch.optim as optim
import dataReader
import os
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import json

from model.KANLinear import KANLinear


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


def evaluate(model, datas, cuda_device, params=["acc"]):
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


class _Hax(nn.Module):
    """T-fixup assumes self-attention norms are removed"""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class Wav2Vec2(nn.Module):
    def __init__(self, feature_encoder, transformer, mask_rate=0.1, mask_span=6, num_negative=20, temp=0.1,
                 l2_power=1.0):
        super(Wav2Vec2, self).__init__()
        self.feature_encoder = feature_encoder
        self.transformer = transformer
        self.mask_rate = mask_rate
        self.mask_span = mask_span
        self.num_negative = num_negative
        self.temp = temp
        self.l2_power = l2_power
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, inputs):
        # 输入特征编码
        # input = [batch, channel, timepoint]
        features = self.feature_encoder(inputs)
        features_unmask = features.clone()

        # 应用掩码
        mask = self._apply_mask(features, self.mask_rate, self.mask_span)

        # 通过 Transformer 生成上下文特征
        context_features = self.transformer(features, mask)

        # 生成负样本,对每个位置都准备了对应的负样本
        negatives, _ = self._generate_negatives(features, self.num_negative)

        # 对比学习
        loss = self._contrastive_similarity(context_features, features_unmask, negatives)

        return loss, features, mask

    def _apply_mask(self, features, mask_rate, mask_span):
        batch_size, dim, seq_len = features.shape
        # mask = [batch, seq_len], a array that saved bool type data to decide which part should be masked
        mask = torch.zeros((batch_size, seq_len), requires_grad=False, dtype=torch.bool)


        for i in range(batch_size):
            mask_seeds = list()
            mask_inds = list()
            while len(mask_seeds) == 0 and mask_rate > 0:
                mask_seeds = np.nonzero(np.random.rand(seq_len) < mask_rate)[0]
                for seed in mask_seeds:
                    for j in range(seed, seed+mask_span):
                        if j >= seq_len:
                            break
                        elif j not in mask_inds:
                            mask_inds.append(j)
            mask[i, mask_inds] = True

        return mask

    def _contrastive_similarity(self, context_features, features_unmask, negatives):
        #   与negatives[batch, seq_len, num_negatives, dim]对齐
        #   [batch, dim, seq_len+1] -> [batch, seq_len, dim] -> [batch, seq_len, 1, dim]
        context_features = context_features[..., 1:].permute([0, 2, 1]).unsqueeze(-2)
        features_unmask = features_unmask.permute([0, 2, 1]).unsqueeze(-2)

        # negative_in_target [batch_size, seq_len, num_negatives] type=bool
        # 去除与正样本相同的负样本
        negative_in_target = (context_features == negatives).all(-1)
        # target [batch_size, seq_len, num_negatives + 1, feat_dim] 第一个为正样本
        targets = torch.cat([context_features, negatives], dim=-2)

        # 计算相似度
        logits = F.cosine_similarity(features_unmask, targets, dim=-1) / self.temp

        if negative_in_target.any():
            logits[1:][negative_in_target] = float("-inf")

        # return [batch_size * seq_len, num_negatives + 1]
        return logits.view(-1, logits.shape[-1])

    def _generate_negatives(self, features, num):
        # 负样本的生成逻辑
        batch, dim, seq_len = features.shape
        # [batch, dim, seq_len] -> [batch*seq_len, dim]
        feature_t = features.permute([0, 2, 1]).reshape(-1, dim)

        with torch.no_grad():
            negative_inds = torch.randint(0, seq_len - 1, size=(batch, seq_len * num))
            for i in range(1, batch):
                negative_inds[i] += i * seq_len

        # final output is [batch, seq_len, num, dim]
        feature_t = feature_t[negative_inds.view(-1)].view(batch, seq_len, num, dim)

        return feature_t, negative_inds

    def calculate_loss(self, outputs):
        logits = outputs[0]
        labels = torch.zeros(logits.shape[0], device=logits.device, dtype=torch.long)
        # Note the loss_fn here integrates the softmax as per the normal classification pipeline (leveraging logsumexp)
        return self.loss_fn(logits, labels) + self.l2_power * outputs[1].pow(2).mean()

    def save(self, filename1, filename2):
        self.feature_encoder.save(filename1)
        self.transformer.save(filename2)


class TransformerEncoder(nn.Module):
    def __init__(self, in_features, n_head=8, dim_feedforward=768, dropout=0.15, activation='gelu',
                 num_layers=6, cls=-5, position_encoder=25):
        super(TransformerEncoder, self).__init__()

        self.infeatures = in_features
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.activation = activation
        self.num_layers = num_layers
        self.cls = cls
        self.position_encoder = position_encoder


        self.cls_token = cls
        # mask replace
        self.mask_replacement = torch.nn.Parameter(torch.normal(0, in_features**(-0.5), size=(in_features,)),
                                                   requires_grad=True)
        # 构建transformer encoder layer
        encoder_layer = nn.TransformerEncoderLayer(d_model=in_features, nhead=n_head, dim_feedforward=dim_feedforward,
                                                   dropout=dropout, activation=activation)
        encoder_layer.norm1 = _Hax()
        encoder_layer.norm1 = _Hax()
        self.encoder_layer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        if position_encoder:
            conv = nn.Conv1d(in_features, in_features, position_encoder, padding=position_encoder // 2, groups=16)
            nn.init.normal_(conv.weight, mean=0, std=2 / in_features)
            nn.init.constant_(conv.bias, 0)
            conv = nn.utils.weight_norm(conv, dim=2)
            self.relative_position = nn.Sequential(conv, nn.GELU())

        self.apply(self.init_bert_params)

    def init_bert_params(self, module):
        if isinstance(module, nn.Linear):
            # module.weight.data.normal_(mean=0.0, std=0.02)
            nn.init.xavier_uniform_(module.weight.data)
            if module.bias is not None:
                module.bias.data.zero_()
            # Tfixup
            module.weight.data = 0.67 * self.num_layers ** (-0.25) * module.weight.data

    def forward(self, x, mask=None):
        # x = [batch, dim, seq_len]
        # print(f'input-{x.shape}')
        if mask is not None:
            x.transpose(2, 1)[mask] = self.mask_replacement

        if self.position_encoder:
            x = x + self.relative_position(x)
        # print(f'add position-{x.shape}')
        # 调整为[seq_length, batch, dim]
        x = x.permute(2, 0, 1)
        # print(f'after permute-{x.shape}')
        if self.cls_token is not None:
            in_token = self.cls_token * torch.ones((1, 1, 1), requires_grad=True).to(x.device).expand(
                [-1, *x.shape[1:]])
            x = torch.cat([in_token, x], dim=0)

        # print(f'add cls-{x.shape}')

        x = self.encoder_layer(x)

        # print(f'output-{x.shape}')
        # print(x.shape)

        return x.permute(1, 2, 0)

    def save(self, filename):
        torch.save(self.state_dict(), filename)

    def load(self, filename):
        parameter = torch.load(filename)
        self.load_state_dict(parameter, strict=False)


class ConvFeatureEncoder(nn.Module):
    def __init__(self, in_channels):
        super(ConvFeatureEncoder, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels, 256, 2, stride=2, padding=1),
            nn.GroupNorm(128, 256),
            nn.GELU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(256, 256, 2, stride=2, padding=1),
            nn.GroupNorm(128, 256),
            nn.GELU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(256, 256, 2, stride=2, padding=1),
            nn.GroupNorm(128, 256),
            nn.GELU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv1d(256, 256, 2, stride=2, padding=1),
            nn.GroupNorm(128, 256),
            nn.GELU()
        )

    def forward(self, x):
        # x = [batch, channel, timePoint]
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        # x = [batch 256, timepoint / 8]
        return x

    def save(self, filename):
        torch.save(self.state_dict(), filename)

    def load(self, filename):
        self.load_state_dict(torch.load(filename))


class transformerTest(nn.Module):
    def __init__(self, convEncoder, transformerencoder, is_kan=False):
        super(transformerTest, self).__init__()
        self.convEncoder = convEncoder
        self.transformerencoder = transformerencoder

        if is_kan:
            # grid_size: int = 5,
            # spline_order: int = 3,
            # scale_noise: float = 0.1,
            # scale_base: float = 1.0,
            # scale_spline: float = 1.0,
            # base_activation = torch.nn.SiLU,
            # grid_eps: float = 0.02,
            # grid_range: tuple = [-1, 1]
            self.classfier = nn.Sequential(KANLinear(
                    in_features=256,
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
            self.classfier = nn.Sequential(nn.Linear(256, 2))

    def forward(self, x):
        x = self.convEncoder(x)
        x = self.transformerencoder(x)
        # print(f'output-{x.shape}')
        # print(x[:, :, -1].shape)
        x = self.classfier(x[:, :, -1])

        return x

    def load_pretrain_parameter(self, conv_encoder_file, transformer_encoder_file, is_freeze=True):
        self.convEncoder.load(conv_encoder_file)
        self.transformerencoder.load(transformer_encoder_file)

        if is_freeze:
            print('------------------freeze------------------------')
            self._frezze_parameter()

        # for name, param in self.named_parameters():
        #     print(name, param.requires_grad)

    def save(self, filename):
        torch.save(self.state_dict(), filename)

    def _frezze_parameter(self):
        for param in self.convEncoder.parameters():
            param.requires_grad = False
        for param in self.transformerencoder.parameters():
            param.requires_grad = False


def test1():
    datasource = './data'
    file_list = sorted(os.listdir(datasource))
    C = 30
    T = 768

    sample_list = []
    for i in range(61, 62):
        print(file_list[i])
        sample_list.extend(dataReader.getEEG(os.path.join(datasource, file_list[i]), '/' + file_list[i],
                                             sample_rate=256))
    sample_list = dataReader.balance(sample_list)
    # it will return no time data
    # fixed: wrongly put the resample optimatal before the events getting
    tmp_list = dataReader.EEGslice(sample_list, slice_length=768, step_length=100, sample_length=768)

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
    train = DataLoader(train, batch_size=4, shuffle=True, drop_last=True)
    valid = DataLoader(valid, batch_size=4, shuffle=True, drop_last=True)
    test = DataLoader(test, batch_size=4, shuffle=True, drop_last=True)

    # parameter
    cuda_device = 0
    loss = nn.CrossEntropyLoss()
    lr = 0.0001


    # net
    spatialConv = ConvFeatureEncoder(30)
    transEncoder = TransformerEncoder(256)
    net = transformerTest(spatialConv, transEncoder)
    net.load_pretrain_parameter('/root/sharedatas/mxg/fetigueDetectionTest/best_convencoder_parameter.pth',
                                '/root/sharedatas/mxg/fetigueDetectionTest/best_transfermor_parameter.pth')

    p_info, p_list = dataReader.see_info(net)
    print(p_info, p_list)

    # print(net)

    net = net.to(cuda_device)
    optimizer = optim.Adam(net.parameters(), lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=20)

    max = 0
    for epoch in range(30):
        net.train()
        running_loss = 0.0

        train_bar = tqdm(train, desc='Training', leave=False)
        train_bar.set_description(f'Epoch-{epoch}')
        for inputs, labels in train_bar:
            optimizer.zero_grad()

            inputs, labels = inputs.to(cuda_device), labels.to(cuda_device)

            outputs = net(inputs)
            # print(outputs,labels)
            l = loss(outputs, labels)
            l.backward()

            optimizer.step()
            scheduler.step()

            running_loss += l.item() / len(train)
            train_bar.set_postfix(loss=running_loss, lr=scheduler.get_last_lr()[0])
        params = ["acc", "auc", "fmeasure"]
        # # params = ["acc"]
        # print(params)
        print("Training Loss ", running_loss)
        print("Train - ", evaluate(net, train, cuda_device, ["acc"]))
        print("Validation - ", evaluate(net, valid, cuda_device, ["acc"]))
        # result = evaluate(net, test, params)
        # if result[0] > max:
        #     max = result[0]
        # # print("Test - ", result)


def test2():
    # pre-train
    data_manager = dataReader.DatasetManager("data-samples-256.npy", 'fatigue_data')
    dataset = dataReader.GetDataset(data_manager.data, 30, 768)
    train = DataLoader(dataset, batch_size=32, shuffle=True, drop_last=True)

    cuda_device = 0
    lr = 0.00001

    spatialConv = ConvFeatureEncoder(30)
    transEncoder = TransformerEncoder(256)
    net = Wav2Vec2(spatialConv, transEncoder)

    net = net.to(cuda_device)
    optimizer = optim.Adam(net.parameters(), lr)

    best_loss = float('inf')
    for epoch in range(80):
        net.train()
        running_loss = 0.0

        train_bar = tqdm(train, desc='Training', leave=False)
        train_bar.set_description(f'Epoch-{epoch}')
        for inputs, _ in train_bar:
            optimizer.zero_grad()

            inputs = inputs.to(cuda_device)

            outputs = net(inputs)
            # print(outputs,labels)
            l = net.calculate_loss(outputs)
            l.backward()

            optimizer.step()
            running_loss += l.item() / len(train)
            train_bar.set_postfix(loss=running_loss)

        if epoch % 5 == 0:
            print(f'Epoch-{epoch}: loss = {running_loss}')

        if running_loss < best_loss:
            best_loss = running_loss
            spatialConv.save('best_convencoder_parameter.pth')
            transEncoder.save('best_transfermor_parameter.pth')


        with open('result_test.json', 'a') as f:
            f.write(f'{epoch}, {running_loss} \n')

def func2():
    data = torch.Tensor(np.random.rand(32, 30, 500))
    spatialConv = ConvFeatureEncoder(30)
    transEncoder = TransformerEncoder(256)
    model = transformerTest(spatialConv, transEncoder, is_kan=True)

    print(model)

    out = model(data)


if __name__ == '__main__':
    # test2()
    # test1()
    func2()
