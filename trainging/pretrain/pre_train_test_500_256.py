from model.KCTSTN import ConvFeatureEncoder, TransformerEncoder, Wav2Vec2
from torch.optim.lr_scheduler import OneCycleLR
import torch.optim as optim
from train_agent import trainAgent

# 不进行下采样
agent_name = 'pre_train_test_500_256'

data_base = {
    'type': 'pre_train',
    'data_source': 'data-samples-500-256.npy',
    'channel': 30,
    'timepoint': 256,
    'sample_rate': 500
}

spatialConv = ConvFeatureEncoder(30)
transEncoder = TransformerEncoder(256, dropout=0.25)
model = Wav2Vec2(spatialConv, transEncoder)

lr = 0.0001
epoch = 80
cuda_settings = {
    'cuda': 1
}
optimizer = optim.Adam
scheduler = OneCycleLR



if __name__ == '__main__':
    # person = []
    # for root, dirs, files in os.walk('/root/sharedatas/mxg/fetigueDetectionTest/data'):
    #     for file in files:
    #         # file_path = os.path.join(root, file)
    #         # print(file_path)
    #         if file.endswith(".fdt"):
    #             continue
    #         person.append(os.path.join(root, file))
    # # print(len(person.keys()))
    #
    data = data_base.copy()
    # data['data_source'] = person

    ta = trainAgent(model, data, agent_name, 'model_saved', optimizer=optimizer, scheduler=scheduler,
                    lr=lr, epoch=epoch, cuda_settings=cuda_settings, batch_size=32)
    ta.training()