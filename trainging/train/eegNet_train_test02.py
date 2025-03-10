from model.eegNet import EEGNet
from torch.optim.lr_scheduler import OneCycleLR
import torch.optim as optim
import torch.nn as nn
from train_agent import trainAgent

import os

# 尝试不使用滑动窗口进行数据增强

agent_name = 'eegNet'

data_base = {
    'type': 'train',
    # 'data_source': ['/root/sharedatas/mxg/fetigueDetectionTest/data/s12_060710_1m.set/s12_060710_1m.set',
    #                 '/root/sharedatas/mxg/fetigueDetectionTest/data/s12_060710_2m.set/s12_060710_2m.set'],
    'channel': 30,
    'timepoint': 500,
    'sample_rate': 500,
    'step_length': 500,
    'feature_record': None,
    'transformer_record': None
}

model = EEGNet()

lr = 0.001
epoch = 100
cuda_settings = {
    'cuda': 0
}
optimizer = optim.Adam
scheduler = OneCycleLR
loss = nn.CrossEntropyLoss()


def func1():
    person = {}
    for root, dirs, files in os.walk('/root/sharedatas/mxg/fetigueDetectionTest/data'):
        for file in files:
            # file_path = os.path.join(root, file)
            # print(file_path)
            if file.endswith(".fdt"):
                continue
            person_mark = file.split('_')[0]
            if person_mark not in person.keys():
                person[person_mark] = [os.path.join(root, file)]
            else:
                person[person_mark].append(os.path.join(root, file))
    # print(len(person.keys()))

    for k in person.keys():
        print(f'person:{k}-------------------------------------------------------------------------------')
        data = data_base.copy()
        data['data_source'] = person[k]
        ta = trainAgent(model, data, k, '../../resluts/eegNet01/228', optimizer=optimizer, scheduler=scheduler, loss=loss,
                        lr=lr, epoch=20, batch_size=32, cuda_settings=cuda_settings)
        ta.training()


def func2():
    person = []
    for root, dirs, files in os.walk('/root/sharedatas/mxg/fetigueDetectionTest/data'):
        for file in files:
            # file_path = os.path.join(root, file)
            # print(file_path)
            if file.endswith(".fdt"):
                continue
            person.append(os.path.join(root, file))
    # print(len(person.keys()))

    data = data_base.copy()
    data['data_source'] = person
    ta = trainAgent(model, data, agent_name, '../../resluts/eegNet01/228', optimizer=optimizer, scheduler=scheduler, loss=loss,
                    lr=lr, epoch=50, batch_size=32, cuda_settings=cuda_settings, freeze=False)
    ta.training()

if __name__ == '__main__':

    func1()
    func2()