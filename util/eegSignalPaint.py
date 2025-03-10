import random
import torch
import mne
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
import os
import tqdm
from typing import Union



def func1(data_source):
    raw = mne.io.read_raw_eeglab(data_source)
    # print(raw.info)

    # 2. 选择 EEG 通道（MNE 数据包含 MEG 和 EEG，我们只取 EEG）
    eeg_picks = mne.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False, exclude='bads')

    # 3. 选择一个特定通道（这里选第一个 EEG 通道）
    selected_channel = eeg_picks[0]
    channel_name = raw.ch_names[selected_channel]

    # 4. 获取所选通道的信号
    time_series, times = raw[selected_channel, :]
    time_series = time_series.squeeze()

    # 5. 绘制所选通道的 EEG 信号并保存
    plt.figure(figsize=(10, 4))
    plt.plot(times[:5000], time_series[:5000], label=channel_name)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (µV)")
    plt.title(f"EEG Signal - {channel_name}")
    plt.savefig(f"EEG_{channel_name}.png")
    plt.show()


if __name__ == '__main__':
    func1('/root/sharedatas/mxg/fetigueDetectionTest/data/dataTest/s01_051017m.set')