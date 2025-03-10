from model.KCTSTN import ConvFeatureEncoder, TransformerEncoder, transformerTest
from torch.optim.lr_scheduler import OneCycleLR
import torch.optim as optim
import torch.nn as nn
from train_agent import trainAgent

import os


agent_name = 'train-by-500-500-anti'
#
# data_base = {
#     'type': 'train',
#     # 'data_source': ['/root/sharedatas/mxg/fetigueDetectionTest/data/s12_060710_1m.set/s12_060710_1m.set',
#     #                 '/root/sharedatas/mxg/fetigueDetectionTest/data/s12_060710_2m.set/s12_060710_2m.set'],
#     'channel': 30,
#     'timepoint': 768,
#     'sample_rate': 256,
#     'feature_record': '/root/sharedatas/mxg/fetigueDetectionTest/pre_train_test_01-convencoder-parameter-11-29-14-25.pth',
#     'transformer_record': '/root/sharedatas/mxg/fetigueDetectionTest/pre_train_test_01-transformer-parameter-11-29-14-25.pth'
# }

data_base = {
    'type': 'train',
    # 'data_source': ['/root/sharedatas/mxg/fetigueDetectionTest/data/s12_060710_1m.set/s12_060710_1m.set',
    #                 '/root/sharedatas/mxg/fetigueDetectionTest/data/s12_060710_2m.set/s12_060710_2m.set'],
    'channel': 30,
    'timepoint': 500,
    'sample_rate': 500,
    'step_length': 100,
    'feature_record': '/root/sharedatas/mxg/fetigueDetectionTest/model_saved/pre_train_test_02_anti-convencoder-parameter-12-25-14-51.pth',
    'transformer_record': '/root/sharedatas/mxg/fetigueDetectionTest/model_saved/pre_train_test_02_anti-transformer-parameter-12-25-14-51.pth'
}

spatialConv = ConvFeatureEncoder(30)
transEncoder = TransformerEncoder(256)
model = transformerTest(spatialConv, transEncoder)

lr = 0.000005
epoch = 50
cuda_settings = {
    'cuda': 0
}
optimizer = optim.Adam
# scheduler = CosineAnnealingLR
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
        try:
            data = data_base.copy()
            data['data_source'] = person[k]
            ta = trainAgent(model, data, k, f'resluts/{agent_name}/forPerson', optimizer=optimizer, scheduler=scheduler, loss=loss,
                            lr=lr, epoch=epoch, cuda_settings=cuda_settings, batch_size=32, freeze=False)
            ta.training()
        except Exception as e:
            with open('./log1218', 'a') as f:
                f.write(f'{k} \n')

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

    lr2 = 0.000005
    data = data_base.copy()
    data['data_source'] = person
    ta = trainAgent(model, data, 'all', f'resluts/{agent_name}', optimizer=optimizer, scheduler=scheduler,
                            loss=loss, lr=lr2, epoch=100, cuda_settings=cuda_settings, batch_size=32, freeze=False)
    ta.training()


if __name__ == '__main__':
    func1()
    # func2()
