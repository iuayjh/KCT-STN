from model.KCTSTN import ConvFeatureEncoder, TransformerEncoder, Wav2Vec2
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.optim as optim
from train_agent import trainAgent


agent_name = 'pre_train_test_01'

data = {
    'type': 'pre_train',
    'data_source': 'data-samples-256.npy',
    'channel': 30,
    'timepoint': 768
}

spatialConv = ConvFeatureEncoder(30)
transEncoder = TransformerEncoder(256)
model = Wav2Vec2(spatialConv, transEncoder)

lr = 0.00001
epoch = 80
cuda_settings = {
    'cuda': 0
}
optimizer = optim.Adam
scheduler = CosineAnnealingLR



if __name__ == '__main__':
    ta = trainAgent(model, data, agent_name, optimizer=optimizer, scheduler=scheduler,
                    lr=lr, epoch=epoch, cuda_settings=cuda_settings)
    ta.training()