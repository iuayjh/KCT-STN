import os.path

import torch.nn

import dataReader
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
from datetime import datetime
from sklearn.metrics import roc_auc_score, precision_score, recall_score, accuracy_score


class trainAgent():
    def __init__(self, model: torch.nn.Module, data: dict, agent_name, result_path, optimizer=None,
                 scheduler=None, loss=None, lr=0.00001, epoch=20, cuda_settings=None, batch_size=4, freeze=True):
        '''
        :param model: the using net
        :param data: a dict {type: pre_train or train, others:}
        :param optimizer:
        :param scheduler:
        :param lr:
        :param epoch:
        :param cuda_settings:
        '''
        self.model = model
        self.data = data
        self.agent_name = agent_name
        self.result_path = result_path
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss = loss
        self.lr = lr
        self.epoch = epoch
        self.cuda_settings = cuda_settings
        self.type = None
        self.batch_size = batch_size
        self.freeze = freeze
        # folder_path = os.path.join(os.getcwd(), self.result_path)
        if not os.path.exists(f'{self.result_path}'):
            os.mkdir(f'{self.result_path}')

        current = datetime.now()
        self.formatted_datetime = current.strftime("%m-%d-%H-%M")

        # checking the basic module
        if data is None:
            raise Exception('no data')

        if model is None:
            raise Exception('no model')

        self._upload_data(self.data)
        print('data upload finished')

    def _upload_data(self, data):
        if 'type' not in data.keys():
            raise Exception('data has no parameter of type')

        if data['type'] == 'pre_train':
            self.type = 'pre_train'
            self._parsing_pretrain_data(data)
        elif data['type'] == 'train':
            self.type = 'train'
            self._parsing_train_data(data)
        else:
            print('no data')

    def _parsing_pretrain_data(self, data):
        self.data_manager = dataReader.DatasetManager(data['data_source'], data['channel'], data['timepoint'],
                                                      data['sample_rate'], 'fatigue_data', type='pre_train')
        self.dataset = dataReader.GetDataset(self.data_manager.data, data['channel'], data['timepoint'])

    def _parsing_train_data(self, data):
        self.data_manager = dataReader.DatasetManager(data['data_source'], data['channel'], data['timepoint'],
                                                      data['sample_rate'], 'fatigue_data', type='train',
                                                      step_length=data['step_length'])
        self.dataset = dataReader.GetDataset_train(self.data_manager.data, data['channel'], data['timepoint'])

    def training(self):
        print(self.type)
        if self.type == 'pre_train':
            self._pre_train()
        elif self.type == 'train':
            self._train()

    def _pre_train(self):

        train = DataLoader(self.dataset, batch_size=32, shuffle=True, drop_last=True)
        cuda_device = self.cuda_settings['cuda']
        model = self.model.to(cuda_device)
        optimizer = self.optimizer(model.parameters(), self.lr)
        scheduler = None
        if self.scheduler:
            # then need to disband the parameter
            # scheduler = self.scheduler(optimizer, T_max=80)
            scheduler = self.scheduler(optimizer, max_lr=self.lr, epochs=self.epoch,
                                       steps_per_epoch=len(train))
        best_loss = float('inf')

        with open(f'{self.result_path}/{self.agent_name}-{self.formatted_datetime}-pre-train-result.json', 'w') as f:
            f.write(f'epoch, running_loss, lr \n')

        for epoch in range(self.epoch):
            model.train()
            running_loss = 0.0

            train_bar = tqdm(train, desc='Training', leave=False)
            train_bar.set_description(f'Epoch-{epoch}')
            for inputs, _ in train_bar:
                optimizer.zero_grad()

                inputs = inputs.to(cuda_device)

                outputs = model(inputs)
                # print(outputs,labels)
                l = model.calculate_loss(outputs)
                l.backward()

                optimizer.step()
                if self.scheduler:
                    scheduler.step()
                lr = scheduler.get_last_lr() if scheduler else self.lr
                running_loss += l.item() / len(train)
                train_bar.set_postfix(loss=running_loss, lr=lr)

            if epoch % 5 == 0:
                print(f'Epoch-{epoch}: loss = {running_loss}')

            # recording
            if running_loss < best_loss:
                best_loss = running_loss
                feature_encoder_name = self.result_path + '/' + self.agent_name + '-' + 'convencoder-parameter' + \
                                       '-' + self.formatted_datetime + '.pth'
                transformer_name = self.result_path + '/' + self.agent_name + '-' + 'transformer-parameter' + \
                                   '-' + self.formatted_datetime + '.pth'

                model.save(feature_encoder_name, transformer_name)

            with open(f'{self.result_path}/{self.agent_name}-{self.formatted_datetime}-pre-train-result.json', 'a') as f:
                f.write(f'{epoch}, {running_loss}, {lr} \n')

    def _train(self):
        with open(f'{self.result_path}/{self.agent_name}-{self.formatted_datetime}-train-result.json', 'w') as f:
            f.write(f'alert:{self.data_manager.alert_sum}, fatigue:{self.data_manager.fatigue_sum}\n')
            f.write(f'epoch, running_loss, lr, train_result, valid_result, test_result \n')

        if self.data_manager.alert_sum <= 2*self.batch_size:
            return
        scheduler = None
        # 划分训练、验证、测试数据集
        train, valid, test = random_split(dataset=self.dataset, lengths=self.data_manager.dataset_split, generator=torch.Generator().manual_seed(0))
        train = DataLoader(train, batch_size=self.batch_size, shuffle=True, drop_last=True)
        valid = DataLoader(valid, batch_size=self.batch_size, shuffle=True, drop_last=True)
        test = DataLoader(test, batch_size=self.batch_size, shuffle=True, drop_last=True)

        # parameter
        cuda_device = self.cuda_settings['cuda']

        # net
        if self.data['feature_record'] and self.data['transformer_record']:
            self.model.load_pretrain_parameter(self.data['feature_record'], self.data['transformer_record'], is_freeze=self.freeze)
        model = self.model.to(cuda_device)
        optimizer = self.optimizer(model.parameters(), self.lr, weight_decay=0.01)
        if self.scheduler:
            scheduler = self.scheduler(optimizer, max_lr=self.lr, epochs=self.epoch,
                                       steps_per_epoch=len(train))

        for epoch in range(self.epoch):
            model.train()
            running_loss = 0.0

            train_bar = tqdm(train, desc='Training')
            train_bar.set_description(f'Epoch-{epoch}')
            for inputs, labels in train_bar:
                optimizer.zero_grad()

                inputs, labels = inputs.to(cuda_device), labels.to(cuda_device)

                outputs = model(inputs)
                # print(outputs,labels)
                l = self.loss(outputs, labels)
                l.backward()

                optimizer.step()
                if self.scheduler:
                    scheduler.step()

                lr = scheduler.get_last_lr() if scheduler else self.lr
                running_loss += l.item() / len(train)
                train_bar.set_postfix(loss=running_loss, lr=lr)

            train_result = evaluate(model, train, cuda_device, ["acc", "recall", "precision", "fmeasure"])
            valid_result = evaluate(model, valid, cuda_device, ["acc", "recall", "precision", "fmeasure"])
            test_result = evaluate(model, test, cuda_device, ["acc", "recall", "precision", "fmeasure"])

            # if running_loss <= 0.55:
            #     self.model.save(f'{self.agent_name}-{self.formatted_datetime}.pth')

            with open(f'{self.result_path}/{self.agent_name}-{self.formatted_datetime}-train-result.json', 'a') as f:
                f.write(f'{epoch}, {running_loss}, {lr}, {train_result}, {valid_result}, {test_result} \n')


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

if __name__ == '__main__':
    current = datetime.now()
    # formatted_datetime = current.strftime("%m-%d-%H-%M")
    # print(formatted_datetime)
