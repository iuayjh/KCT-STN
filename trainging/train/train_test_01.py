from model.KCTSTN import ConvFeatureEncoder, TransformerEncoder, transformerTest
from torch.optim.lr_scheduler import OneCycleLR
import torch.optim as optim
import torch.nn as nn
from train_agent import trainAgent


agent_name = 'train_test_01'

data = {
    'type': 'train',
    'data_source': ['/root/sharedatas/mxg/fetigueDetectionTest/data/s12_060710_1m.set/s12_060710_1m.set',
                    '/root/sharedatas/mxg/fetigueDetectionTest/data/s12_060710_2m.set/s12_060710_2m.set'],
    'channel': 30,
    'timepoint': 768,
    'sample_rate': 500,
    'feature_record': '/root/sharedatas/mxg/fetigueDetectionTest/pre_train_test_01-convencoder-parameter-11-29-14-25.pth',
    'transformer_record': '/root/sharedatas/mxg/fetigueDetectionTest/pre_train_test_01-transformer-parameter-11-29-14-25.pth'
}

spatialConv = ConvFeatureEncoder(30)
transEncoder = TransformerEncoder(256)
model = transformerTest(spatialConv, transEncoder)

lr = 0.00001
epoch = 20
cuda_settings = {
    'cuda': 0
}
optimizer = optim.Adam
scheduler = OneCycleLR
loss = nn.CrossEntropyLoss()


if __name__ == '__main__':
    ta = trainAgent(model, data, agent_name, '../../resluts', optimizer=optimizer, scheduler=scheduler, loss=loss,
                    lr=lr, epoch=epoch, cuda_settings=cuda_settings, freeze=False)
    ta.training()