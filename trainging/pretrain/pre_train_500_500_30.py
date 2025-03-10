from model.KCTSTN import ConvFeatureEncoder, TransformerEncoder, Wav2Vec2
from torch.optim.lr_scheduler import OneCycleLR
import torch.optim as optim
from train_agent import trainAgent

# 不进行下采样
agent_name = 'pre_train_500_500_30'

data_base = {
    'type': 'pre_train',
    'data_source': 'tmp_data/data-samples-500-500_30%.npy',
    'channel': 30,
    'timepoint': 500,
    'sample_rate': 500
}

spatialConv = ConvFeatureEncoder(30)
transEncoder = TransformerEncoder(256, dropout=0.25)
model = Wav2Vec2(spatialConv, transEncoder)

lr = 0.0001
epoch = 80
cuda_settings = {
    'cuda': 0
}
optimizer = optim.Adam
scheduler = OneCycleLR



if __name__ == '__main__':
    data = data_base.copy()

    ta = trainAgent(model, data, agent_name, 'model_saved/less_pre_data', optimizer=optimizer, scheduler=scheduler,
                    lr=lr, epoch=epoch, cuda_settings=cuda_settings, batch_size=32)
    ta.training()
