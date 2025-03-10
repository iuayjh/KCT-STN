import os
import json
import pandas as pd


def extract_best_results(input_folder, output_file):
    """
    从文件夹中所有 JSON 文件中提取精度最高的数据并保存到 Excel 文件中。

    :param input_folder: 存储 JSON 文件的文件夹路径。
    :param output_file: 输出的 Excel 文件路径。
    """
    # 初始化结果列表
    results = []

    # 遍历文件夹中的所有 JSON 文件
    for file_name in os.listdir(input_folder):
        if file_name.endswith(".json"):  # 只处理 JSON 文件
            # print(file_name)
            file_path = os.path.join(input_folder, file_name)
            # print(file_name)
            data = []
            # 读取 JSON 文件
            with open(file_path, "r") as f:
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
                training_data['train_result'] = training_data['train_result'].astype('float32')
                training_data['train_recall'] = training_data['train_recall'].astype('float32')
                training_data['train_precision'] = training_data['train_precision'].astype('float32')
                training_data['train_fmeasure'] = training_data['train_fmeasure'].astype('float32')

                training_data['valid_result'] = training_data['valid_result'].astype('float32')
                training_data['valid_recall'] = training_data['valid_recall'].astype('float32')
                training_data['valid_precision'] = training_data['valid_precision'].astype('float32')
                training_data['valid_fmeasure'] = training_data['valid_fmeasure'].astype('float32')

                training_data['test_result'] = training_data['test_result'].astype('float32')
                training_data['test_recall'] = training_data['test_recall'].astype('float32')
                training_data['test_precision'] = training_data['test_precision'].astype('float32')
                training_data['test_fmeasure'] = training_data['test_fmeasure'].astype('float32')


            else:
                print(f"文件 {file_name} 格式错误或无有效记录，已跳过。")
                continue

            # print(training_data)

            # 找到精度最高的数据（根据 valid_result 的第一个值）
            try:
                best_epoch = training_data.loc[29]
                # best_epoch = training_data.loc[training_data['valid_result'].idxmax()]
                # print(best_epoch)
            except Exception as e:
                pass
                # best_epoch = training_data.loc[training_data['epoch']==19]
                # best_epoch = training_data.loc[training_data['valid_result'].idxmax()]
            best_epoch = best_epoch.to_dict()
            best_epoch['person'] = file_name.split('-')[0]  # 添加文件名信息
            best_epoch['alert'] = int(count[0][6:])
            best_epoch['fatigue'] = int(count[1][8:])
            # print(best_epoch)
            results.append(best_epoch)

    # 检查是否有有效结果
    if not results:
        print("未找到任何有效的训练记录。")
        return

    # 将结果转换为 DataFrame
    df = pd.DataFrame(results)

    cols = ['person'] + [col for col in df.columns if col != 'person']
    df = df[cols]

    df = df.sort_values(by='person')

    # 保存到 Excel 文件
    df.to_excel(output_file, index=False)
    print(f"统计结果已保存到: {output_file}")


# 示例使用
input_folder = "/root/sharedatas/mxg/fetigueDetectionTest/resluts/CNNLSTM-500-256/kanTest-228"  # 替换为存储 JSON 文件的文件夹路径
output_file = "/root/sharedatas/mxg/fetigueDetectionTest/resluts/CNNLSTM-500-256/kanTest-228/kcl-stn-modify-results.xlsx"  # 输出的 Excel 文件名
extract_best_results(input_folder, output_file)
