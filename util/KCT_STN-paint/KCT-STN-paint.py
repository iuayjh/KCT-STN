import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# plt.rcParams['axes.facecolor'] = 'gray'

def paint_loss(dataSource):
    data = []
    # 读取 JSON 文件
    with open(dataSource, "r") as f:
        for idx, line in enumerate(f.readlines()):
            if idx == 0:
                signal_line = line[:-1].split(', ')
            else:
                signal_line = line[:-2].split(', ')
            signal_line = [s.strip('[]') for s in signal_line]
            # print(signal_line)
            data.append(signal_line)
    # print(data)
    # 跳过文件前两行非训练记录部分
    if isinstance(data, list) and len(data) > 2:
        count = data[0]
        # ['acc', 'recall', 'precision', 'fmeasure']
        # columns = data[1]
        columns = ['epoch', 'running_loss', 'lr', 'train_result', 'train_recall', 'train_precision', 'train_fmeasure',
                   'valid_result', 'valid_recall', 'valid_precision', 'valid_fmeasure',
                   'test_result', 'test_recall', 'test_precision', 'test_fmeasure']
        training_data = pd.DataFrame(data=data[2:], columns=columns)  # 从第3行开始处理
        training_data['epoch'] = training_data['epoch'].astype('int32')
        training_data['running_loss'] = training_data['running_loss'].astype('float32')

        training_data['train_result'] = training_data['train_result'].astype('float32')
        # training_data['train_recall'] = training_data['train_recall'].astype('float32')
        # training_data['train_precision'] = training_data['train_precision'].astype('float32')
        # training_data['train_fmeasure'] = training_data['train_fmeasure'].astype('float32')

        training_data['valid_result'] = training_data['valid_result'].astype('float32')
        # training_data['valid_recall'] = training_data['valid_recall'].astype('float32')
        # training_data['valid_precision'] = training_data['valid_precision'].astype('float32')
        # training_data['valid_fmeasure'] = training_data['valid_fmeasure'].astype('float32')

        training_data['test_result'] = training_data['test_result'].astype('float32')
        # training_data['test_recall'] = training_data['test_recall'].astype('float32')
        # training_data['test_precision'] = training_data['test_precision'].astype('float32')
        # training_data['test_fmeasure'] = training_data['test_fmeasure'].astype('float32')

        training_data = training_data.dropna()

    # print(training_data["running_loss"])
    # epoch_list = training_data["epoch"].tolist()
    # running_loss_list = training_data["running_loss"].tolist()

    plt.figure(figsize=(8, 5))  # 设置图表大小
    plt.plot(training_data["epoch"], training_data["running_loss"], linestyle='-', color='b',
             label="Train Loss")

    # 设置标题和坐标轴标签
    plt.title("Training Loss Over Epochs", fontsize=14)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)

    # 设置 x 轴刻度：从 0 到 50，每隔 5 显示一次
    plt.xticks(np.arange(0, 101, 10))

    # 设置 y 轴刻度：从 0 到 1，每隔 0.2 显示一次
    plt.yticks(np.arange(0, 1.0, 0.1))

    # 添加网格
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.legend()  # 显示图例

    # 保存图表为 PNG 格式
    plt.savefig("kst-training_loss_plot.png", dpi=300, bbox_inches='tight')

    plt.show()  # 显示图表


def paint_train_acc(dataSource):
    data = []
    # 读取 JSON 文件
    with open(dataSource, "r") as f:
        for idx, line in enumerate(f.readlines()):
            if idx == 0:
                signal_line = line[:-1].split(', ')
            else:
                signal_line = line[:-2].split(', ')
            signal_line = [s.strip('[]') for s in signal_line]
            # print(signal_line)
            data.append(signal_line)
    # print(data)
    # 跳过文件前两行非训练记录部分
    if isinstance(data, list) and len(data) > 2:
        count = data[0]
        # ['acc', 'recall', 'precision', 'fmeasure']
        # columns = data[1]
        columns = ['epoch', 'running_loss', 'lr', 'train_result', 'train_recall', 'train_precision', 'train_fmeasure',
                   'valid_result', 'valid_recall', 'valid_precision', 'valid_fmeasure',
                   'test_result', 'test_recall', 'test_precision', 'test_fmeasure']
        training_data = pd.DataFrame(data=data[2:], columns=columns)  # 从第3行开始处理
        training_data['epoch'] = training_data['epoch'].astype('int32')
        training_data['running_loss'] = training_data['running_loss'].astype('float32')

        training_data['train_result'] = training_data['train_result'].astype('float32')
        # training_data['train_recall'] = training_data['train_recall'].astype('float32')
        # training_data['train_precision'] = training_data['train_precision'].astype('float32')
        # training_data['train_fmeasure'] = training_data['train_fmeasure'].astype('float32')

        training_data['valid_result'] = training_data['valid_result'].astype('float32')
        # training_data['valid_recall'] = training_data['valid_recall'].astype('float32')
        # training_data['valid_precision'] = training_data['valid_precision'].astype('float32')
        # training_data['valid_fmeasure'] = training_data['valid_fmeasure'].astype('float32')

        training_data['test_result'] = training_data['test_result'].astype('float32')
        # training_data['test_recall'] = training_data['test_recall'].astype('float32')
        # training_data['test_precision'] = training_data['test_precision'].astype('float32')
        # training_data['test_fmeasure'] = training_data['test_fmeasure'].astype('float32')

        training_data = training_data.dropna()

    # print(training_data["running_loss"])
    # epoch_list = training_data["epoch"].tolist()
    # running_loss_list = training_data["running_loss"].tolist()

    plt.figure(figsize=(8, 5))  # 设置图表大小
    plt.plot(training_data["epoch"], training_data["train_result"], linestyle='-', color='b',
             label="Train ACC")
    plt.plot(training_data["epoch"], training_data["test_result"], linestyle='-', color='g',
             label="Test ACC")

    # 设置标题和坐标轴标签
    plt.title("ACC Over Epochs", fontsize=14)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("ACC", fontsize=12)

    # 设置 x 轴刻度：从 0 到 50，每隔 5 显示一次
    plt.xticks(np.arange(0, 101, 10))

    # 设置 y 轴刻度：从 0 到 1，每隔 0.2 显示一次
    plt.yticks(np.arange(0.5, 1.0, 0.05))

    # 添加网格
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.legend()  # 显示图例

    # 保存图表为 PNG 格式
    plt.savefig("kst-training_acc_plot.png", dpi=300, bbox_inches='tight')

    plt.show()  # 显示图表


def paint_noAug_train_acc(dataSource):
    data = []
    # 读取 JSON 文件
    with open(dataSource, "r") as f:
        for idx, line in enumerate(f.readlines()):
            if idx == 0:
                signal_line = line[:-1].split(', ')
            else:
                signal_line = line[:-2].split(', ')
            signal_line = [s.strip('[]') for s in signal_line]
            # print(signal_line)
            data.append(signal_line)
    # print(data)
    # 跳过文件前两行非训练记录部分
    if isinstance(data, list) and len(data) > 2:
        # ['acc', 'recall', 'precision', 'fmeasure']
        columns = data[0]
        # columns = ['epoch', 'running_loss', 'lr', 'train_result', 'train_recall', 'train_precision', 'train_fmeasure',
        #            'valid_result', 'valid_recall', 'valid_precision', 'valid_fmeasure',
        #            'test_result', 'test_recall', 'test_precision', 'test_fmeasure']
        training_data = pd.DataFrame(data=data[1:], columns=columns)  # 从第3行开始处理
        training_data['epoch'] = training_data['epoch'].astype('int32')
        training_data['running_loss'] = training_data['running_loss'].astype('float32')

        training_data = training_data.dropna()

    # print(training_data["running_loss"])
    # epoch_list = training_data["epoch"].tolist()
    # running_loss_list = training_data["running_loss"].tolist()

    plt.figure(figsize=(8, 5))  # 设置图表大小
    plt.plot(training_data["epoch"], training_data["running_loss"], linestyle='-', color='b',
             label="Train loss")


    # 设置标题和坐标轴标签
    plt.title("Loss Over Epochs", fontsize=14)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("loss", fontsize=12)

    # 设置 x 轴刻度：从 0 到 50，每隔 5 显示一次
    plt.xticks(np.arange(0, 71, 10))

    # 设置 y 轴刻度：从 0 到 1，每隔 0.2 显示一次
    plt.yticks(np.arange(0.0, 1.2, 0.1))

    # 添加网格
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.legend()  # 显示图例

    # 保存图表为 PNG 格式
    plt.savefig("kct-pretraining_loss_plot.png", dpi=300, bbox_inches='tight')

    plt.show()  # 显示图表

if __name__ == '__main__':
    # dataSource = '/root/sharedatas/mxg/fetigueDetectionTest/resluts/train-by-500-500-anti/moreEpoch/all-01-10-17-11-train-result.json'
    dataSource = '/root/sharedatas/mxg/fetigueDetectionTest/resluts/train-by-500-500/normal_test228/all-02-28-17-17-train-result.json'

    # paint_loss(dataSource)
    # paint_train_acc(dataSource)

    # pre_data = '/root/sharedatas/mxg/fetigueDetectionTest/model_saved/pre_train_test_02_anti-12-25-14-51-pre-train-result.json'
    pre_data = '/root/sharedatas/mxg/fetigueDetectionTest/model_saved/pre_train_test_02-12-25-12-02-pre-train-result.json'
    paint_noAug_train_acc(pre_data)