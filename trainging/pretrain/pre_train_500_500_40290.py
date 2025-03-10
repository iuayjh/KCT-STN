from model.KCTSTN import ConvFeatureEncoder, TransformerEncoder, Wav2Vec2
from torch.optim.lr_scheduler import OneCycleLR
import torch.optim as optim
from train_agent import trainAgent
base_root = '/root/sharedatas/mxg/fetigueDetectionTest'

# 不进行下采样
agent_name_base = 'pre_train_500_500_{}'
data_source_base = base_root + '/tmp_data/data-samples-500-500_{}%.npy'

data_base = {
    'type': 'pre_train',
    # 'data_source': 'tmp_data/data-samples-500-500_30%.npy',
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
    num_list = [40, 50, 60, 70, 80, 90]
    for n in num_list:
        agent_name = agent_name_base.format(n)
        data_source = data_source_base.format(n)
        print(f'--------------------------------------{agent_name}--------------------------------------')
        # print(data_source)

        data = data_base.copy()
        data['data_source'] = data_source

        ta = trainAgent(model, data, agent_name, f'{base_root}/model_saved/less_pre_data', optimizer=optimizer,
                        scheduler=scheduler, lr=lr, epoch=epoch, cuda_settings=cuda_settings, batch_size=32)
        ta.training()
