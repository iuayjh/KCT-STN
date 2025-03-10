import random
import torch
import mne
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
import os
import tqdm
from typing import Union

EEG_20_div = [
            'FP1', 'FP2',
    'F7', 'F3', 'FZ', 'F4', 'F8',
    'FT7', 'FC3', 'FCZ', 'FC4', 'FT8',
    'T5', 'P3', 'PZ', 'P4', 'T6',
            'O1', 'OZ', 'O2'
]

def see_info(model: torch.nn.Module):
    """
       输入一个PyTorch Model对象，返回模型的总参数量（格式化为易读格式）以及每一层的名称、尺寸、精度、参数量、是否可训练和层的类别。

       :param model: PyTorch Model
       :return: (总参数量信息, 参数列表[包括每层的名称、尺寸、数据类型、参数量、是否可训练和层的类别])
       """
    params_list = []
    total_params = 0
    total_params_non_trainable = 0

    for name, param in model.named_parameters():
        # 获取参数所属层的名称
        layer_name = name.split('.')[0]
        # 获取层的对象
        layer = dict(model.named_modules())[layer_name]
        # 获取层的类名
        layer_class = layer.__class__.__name__

        params_count = param.numel()
        trainable = param.requires_grad
        params_list.append({
            'tensor': name,
            'layer_class': layer_class,
            'shape': str(list(param.size())),
            'precision': str(param.dtype).split('.')[-1],
            'params_count': str(params_count),
            'trainable': str(trainable),
        })
        total_params += params_count
        if not trainable:
            total_params_non_trainable += params_count

    total_params_trainable = total_params - total_params_non_trainable

    def format_size(size):
        # 对总参数量做格式优化
        K, M, B = 1e3, 1e6, 1e9
        if size == 0:
            return '0'
        elif size < M:
            return f"{size / K:.1f}K"
        elif size < B:
            return f"{size / M:.1f}M"
        else:
            return f"{size / B:.1f}B"

    total_params_info = {
        'total_params': format_size(total_params),
        'total_params_trainable': format_size(total_params_trainable),
        'total_params_non_trainable': format_size(total_params_non_trainable)
    }

    return total_params_info, params_list


class GetDataset(torch.utils.data.Dataset):
    def __init__(self, data_list, C, T):
        super(GetDataset, self).__init__()
        self.data_list = data_list
        self.C = C
        self.T = T

    def __getitem__(self, index):
        datas = self.data_list[index].reshape(self.C, self.T).astype('float32')
        labels = 0
        return datas, labels

    def __len__(self):
        return len(self.data_list)

class GetDataset_train(torch.utils.data.Dataset):
    def __init__(self, data_list, C, T):
        super(GetDataset_train, self).__init__()
        self.data_list = data_list
        self.C = C
        self.T = T

    def __getitem__(self, index):
        datas = self.data_list[index][0].reshape(self.C, self.T).astype('float32')
        labels = self.data_list[index][1]
        return datas, labels

    def __len__(self):
        return len(self.data_list)

class DatasetManager:
    # a class for manage dataset
    def __init__(self, data_source: Union[list, str], channels, timepoint, sample_rate, dataset_name=None, step_length=100, type='train'):
        super(DatasetManager, self).__init__()
        self.data_source = data_source
        self.dataset_name = dataset_name
        self.channels = channels
        self.timepoint = timepoint
        self.sample_rate = sample_rate
        self.type = type
        self.data = None
        self.step_length = step_length
        if isinstance(data_source, str):
            if data_source.split('.')[-1] == 'npy':
                self.data = np.load(self.data_source)
                # print(self.data.shape)
            else:
                self._parsing_from_source(self.data_source)
        else:
            self._parsing_from_source(self.data_source, root=False)

    def _parsing_from_source(self, data_source, root=True):
        """
        :param data_source:
        :param slice_time:  the time decide the length of a slice
        :param sample_rate:
        :return:
        """
        set_files = []
        if root:
            for root, dirs, files in os.walk(data_source):
                for file in files:
                    if file.endswith(".set"):
                        # 获取完整路径
                        full_path = os.path.join(root, file)
                        set_files.append(full_path)
            set_files = sorted(set_files)
        else:
            set_files = data_source
        # print(set_files)
        if self.type == 'pre_train':
            bar = tqdm.tqdm(set_files, desc=f'parsing {self.dataset_name} data')
            for file in bar:
                bar.set_postfix(data_len=0 if self.data is None else self.data.shape[0])
                self._eeg_parser(file, self.timepoint, sample_rate=self.sample_rate)
        elif self.type == 'train':
            self._eeg_parser_train(set_files)
            self._train_dataset_parameters()
        else:
            pass

    def test(self):
        # self._parsing_from_source(self.data_source)
        np.save("data-samples-500-500.npy", self.data)

    def _eeg_parser(self, file, timepoint, sample_rate=256):
        raw = mne.io.read_raw_eeglab(file)

        if raw.info['sfreq'] != sample_rate:
            # print('different sfreq')
            raw.resample(sample_rate)

        # seq_len = sample_rate * slice_time
        seq_len = timepoint
        total_duration = raw.times[-1]

        # 初始化一个列表来存储切片
        slices = []

        # 使用滑动窗口长切分数据
        for start in range(0, int(total_duration * sample_rate), seq_len):
            stop = start + seq_len
            if stop > len(raw.times):
                break  # 如果到达最后一个片段，退出循环
            slice_data = raw[:, start:stop][0]  # 提取时间段的 EEG 数据
            slices.append(slice_data)

        tmp_arr = np.array(slices)
        if self.data is None:
            self.data = tmp_arr
        else:
            self.data = np.concatenate((self.data, tmp_arr), axis=0)

    def _eeg_parser_train(self, files):
        sample_list = []
        bar = tqdm.tqdm(files, desc=f'parsing {self.dataset_name} data')
        for data in bar:
            sample_list.extend(getEEG(data, sample_rate=self.sample_rate))
        # self.data = sample_list
        sample_list = balance(sample_list)
        # self.data = sample_list
        self.data = EEGslice(sample_list, slice_length=self.timepoint, step_length=self.step_length,
                             sample_length=3*self.sample_rate)

    def _train_dataset_parameters(self):
        a = len(self.data)
        l = [0.6, 0.2, 0.2]
        for i in range(len(l)):
            l[i] = int(l[i] * a)
        l[2] = a - l[0] - l[1]

        self.dataset_split = l

        alert, fatigue = 0, 0
        for it in self.data:
            if it[1] == 0:
                fatigue += 1
            else:
                alert += 1
        self.alert_sum = alert
        self.fatigue_sum = fatigue
        print(f'alert:{self.alert_sum} fatigue:{self.fatigue_sum}')




def getEEG(data_source, time=3, sample_rate=500, is_fliter=False):
    """
    通过路径获得标记的脑电信号，脑电信号均为发生车道偏移（事件251/252）前3s
    :param sample_rate: 500
    :param time: 3
    :param data_source:
    :param file_name:
    :return: a list [(channel eeg sign,mark)]
    """
    raw = mne.io.read_raw_eeglab(data_source)
    if is_fliter:
        raw = raw.pick_channels(EEG_20_div)
    if raw.info['sfreq'] != sample_rate:
        # print('different sfreq')
        raw.resample(sample_rate)

    events = mne.events_from_annotations(raw, event_id={
        '251': 251,
        '252': 252,
        '253': 253,
        '254': 254
    })

    # 切出RT时间
    time_list = []
    for i in range(len(events[0])):
        start, end = 0, 0
        if events[0][i][2] == 254:
            continue
        elif events[0][i][2] == 253:
            continue
        elif events[0][i][2] == 251 or events[0][i][2] == 252:
            if events[0][i + 1][2] == 253:
                start = events[0][i][0]
                end = events[0][i + 1][0]
                time_list.append([start, end])

    # 计算RT时间
    time_spend = []
    for t in time_list:
        tmp = t[1] - t[0]
        if 0.2 * sample_rate < tmp < 30 * sample_rate:
            time_spend.append(tmp)
    alert_RT = np.percentile(time_spend, 5)

    # 标记疲劳情况
    # 见 Inter-subject transfer learning for EEG-based mental fatigue recognition
    #    Toward Drowsiness Detection Using Non-hair Bearing EEG-Based Brain-Computer Interfaces
    # 0-fatigue, 1-alert, 2-other, 3-wrong
    fatigue_mark = []
    global_index = 90 * sample_rate
    for i in range(len(time_list)):
        local_start = time_list[i][0] - global_index
        local_RT = time_list[i][1] - time_list[i][0]
        if 0.2*sample_rate < local_RT < 30*sample_rate:
            global_RT = 0
            global_count = 0
            for j in range(i):
                tmp = time_list[j][1] - time_list[j][0]
                if time_list[j][0] >= local_start:
                    global_RT += tmp
                    global_count += 1
            if global_count > 0:
                global_RT = global_RT / global_count
            else:
                global_RT = local_RT
            if local_RT <= 1.5 * alert_RT and global_RT <= 1.5 * alert_RT:
                fatigue_mark.append(1)
            elif local_RT >= 2.5 * alert_RT and global_RT >= 2.5 * alert_RT:
                fatigue_mark.append(0)
            else:
                fatigue_mark.append(2)
        else:
            fatigue_mark.append(3)
    sample_label = []
    for i in range(len(fatigue_mark)):
        # alert
        if fatigue_mark[i] == 1:
            data, _ = raw[:, time_list[i][0] - time * sample_rate:time_list[i][0]]
            sample_label.append((data, 1))
        # fatigue
        elif fatigue_mark[i] == 0:
            data, _ = raw[:, time_list[i][0] - time * sample_rate:time_list[i][0]]
            sample_label.append((data, 0))

    return sample_label


def EEGslice(sample_list, slice_length=200, step_length=100, sample_length=1500):
    """
    对eeg信号进行切片
    :param sample_length: time * sample_rate
    :param step_length:
    :param sample_list: target list
    :param slice_length: the length of slice
    :return: a list [(channel eeg sign,mark)]
    """
    tmp_list = []
    for it in sample_list:
        for i in range(int((sample_length - slice_length + step_length) / step_length)):
            start = step_length * i
            if start + slice_length <= sample_length:
                A = np.copy(it[0][:, start:start + slice_length])
                tmp_list.append((A, it[1]))

    return tmp_list


def balance(sample_list):
    alert, fatigue = 0, 0
    for it in sample_list:
        if it[1] == 0:
            fatigue += 1
        else:
            alert += 1

    new_list = []
    if alert > fatigue:
        for it in sample_list:
            if it[1] == 0:
                new_list.append(it)
            else:
                if random.random() <= fatigue / alert:
                    new_list.append(it)

        return new_list
    elif alert < fatigue:
        for it in sample_list:
            if it[1] == 0:
                if random.random() <= alert / fatigue:
                    new_list.append(it)
            else:
                new_list.append(it)

        return new_list
    else:
        return sample_list


def test():
    datasource = './data'
    file_list = sorted(os.listdir(datasource))
    C = 20
    T = 200

    sample_list = []
    for i in range(61, 62):
        sample_list.extend(getEEG(os.path.join(datasource, file_list[i]), '/' + file_list[i]))
        sample_list = balance(sample_list)
        tmp_list = EEGslice(sample_list)


if __name__ == '__main__':
    # test()
    # dataset = DatasetManager('/root/sharedatas/mxg/fetigueDetectionTest/data', 30, 500,
    #                                                   500, 'fatigue_data', type='pre_train')
    # # dataset = DatasetManager("data-samples-256.npy", 'fatigue_data')
    # dataset.test()

    # dataset = DatasetManager('/root/sharedatas/mxg/fetigueDetectionTest/data', 30, 500,
    #                                                   500, 'fatigue_data', type='train')
    # dataset = DatasetManager("data-samples-256.npy", 'fatigue_data')

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

    for k in person.keys():
        print(f'person:{k}-------------------------------------------------------------------------------')

        dataset = DatasetManager(person[k], 30, 500, 500, 'fatigue_data', type='train')







