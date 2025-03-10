from model.KCTSTN import ConvFeatureEncoder, TransformerEncoder, transformerTest
from torch.optim.lr_scheduler import OneCycleLR
import torch.optim as optim
import torch.nn as nn
from train_agent import trainAgent

import os
base_root = '/root/sharedatas/mxg/fetigueDetectionTest'

agent_name_base = 'train-by-500-500-{}'


data_base = {
    'type': 'train',
    'channel': 30,
    'timepoint': 500,
    'sample_rate': 500,
    'step_length': 100,
    # 'feature_record': '/root/sharedatas/mxg/fetigueDetectionTest/model_saved/pre_train_test_02-convencoder-parameter-12-25-12-02.pth',
    # 'transformer_record': '/root/sharedatas/mxg/fetigueDetectionTest/model_saved/pre_train_test_02-transformer-parameter-12-25-12-02.pth'
}

spatialConv = ConvFeatureEncoder(30)
transEncoder = TransformerEncoder(256, dropout=0.25)
model = transformerTest(spatialConv, transEncoder)

lr = 0.000005
epoch = 50
cuda_settings = {
    'cuda': 1
}
optimizer = optim.Adam
# scheduler = CosineAnnealingLR
scheduler = OneCycleLR
loss = nn.CrossEntropyLoss()

def func1(agent_name):
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

def func2(agent_name, feature_record, transformer_record):
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
    data['feature_record'] = feature_record
    data['transformer_record'] = transformer_record
    ta = trainAgent(model, data, 'all', f'{base_root}/resluts/{agent_name}', optimizer=optimizer, scheduler=scheduler,
                            loss=loss, lr=lr2, epoch=100, cuda_settings=cuda_settings, batch_size=32, freeze=False)
    ta.training()


if __name__ == '__main__':
    num_list = ['10', '20', '30', '40', '50', '60', '70', '80', '90']
    model_dict = {
        '10': ['/root/sharedatas/mxg/fetigueDetectionTest/model_saved/less_pre_data/pre_train_500_500_10-convencoder-parameter-01-02-13-55.pth',
               '/root/sharedatas/mxg/fetigueDetectionTest/model_saved/less_pre_data/pre_train_500_500_10-transformer-parameter-01-02-13-55.pth'],
        '20': ['/root/sharedatas/mxg/fetigueDetectionTest/model_saved/less_pre_data/pre_train_500_500_20-convencoder-parameter-01-02-14-37.pth',
               '/root/sharedatas/mxg/fetigueDetectionTest/model_saved/less_pre_data/pre_train_500_500_20-transformer-parameter-01-02-14-37.pth'],
        '30': ['/root/sharedatas/mxg/fetigueDetectionTest/model_saved/less_pre_data/pre_train_500_500_30-convencoder-parameter-01-02-16-00.pth',
               '/root/sharedatas/mxg/fetigueDetectionTest/model_saved/less_pre_data/pre_train_500_500_30-transformer-parameter-01-02-16-00.pth'],
        '40': ['/root/sharedatas/mxg/fetigueDetectionTest/model_saved/less_pre_data/pre_train_500_500_40-convencoder-parameter-01-03-13-28.pth',
               '/root/sharedatas/mxg/fetigueDetectionTest/model_saved/less_pre_data/pre_train_500_500_40-transformer-parameter-01-03-13-28.pth'],
        '50': ['/root/sharedatas/mxg/fetigueDetectionTest/model_saved/less_pre_data/pre_train_500_500_50-convencoder-parameter-01-03-16-04.pth',
               '/root/sharedatas/mxg/fetigueDetectionTest/model_saved/less_pre_data/pre_train_500_500_50-transformer-parameter-01-03-16-04.pth'],
        '60': ['/root/sharedatas/mxg/fetigueDetectionTest/model_saved/less_pre_data/pre_train_500_500_60-convencoder-parameter-01-03-19-16.pth',
               '/root/sharedatas/mxg/fetigueDetectionTest/model_saved/less_pre_data/pre_train_500_500_60-transformer-parameter-01-03-19-16.pth'],
        '70': ['/root/sharedatas/mxg/fetigueDetectionTest/model_saved/less_pre_data/pre_train_500_500_70-convencoder-parameter-01-03-23-10.pth',
               '/root/sharedatas/mxg/fetigueDetectionTest/model_saved/less_pre_data/pre_train_500_500_70-transformer-parameter-01-03-23-10.pth'],
        '80': ['/root/sharedatas/mxg/fetigueDetectionTest/model_saved/less_pre_data/pre_train_500_500_80-convencoder-parameter-01-04-03-43.pth',
               '/root/sharedatas/mxg/fetigueDetectionTest/model_saved/less_pre_data/pre_train_500_500_80-transformer-parameter-01-04-03-43.pth'],
        '90': ['/root/sharedatas/mxg/fetigueDetectionTest/model_saved/less_pre_data/pre_train_500_500_90-convencoder-parameter-01-04-08-53.pth',
               '/root/sharedatas/mxg/fetigueDetectionTest/model_saved/less_pre_data/pre_train_500_500_90-transformer-parameter-01-04-08-53.pth']
    }
    for n in num_list:
        agent_name = agent_name_base.format(n)
        func2(agent_name, model_dict[n][0], model_dict[n][1])


