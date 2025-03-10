import pandas as pd
import matplotlib.pyplot as plt
import os


def compare_excel_files(file_dict, output_path, compare_column='train_result', compare_person=None, exlude=['s48']):
    """
    对多个 Excel 文件中 person 字段相同的数据进行对比，并生成一张柱状图。

    :param file_list: 包含 Excel 文件路径的列表。
    :param output_file: 输出图表的文件路径。
    :param compare_column: 要参与比较的列名。
    """
    # 初始化结果存储
    combined_data = {}
    file_list = list(file_dict.keys())
    files = list(file_dict.values())
    # 遍历 Excel 文件列表
    for file_path in file_list:
        file_name = os.path.basename(file_path)  # 提取文件名
        df = pd.read_excel(file_path)

        # 确保必要字段存在
        if 'person' not in df.columns or compare_column not in df.columns:
            print(f"文件 {file_name} 缺少必要的字段，已跳过。")
            continue

        # 记录数据来源及每个 person 的对应值
        for _, row in df.iterrows():
            person = row['person']
            if person not in combined_data:
                combined_data[person] = {}
            combined_data[person][file_dict[file_path]] = row[compare_column]

    # 获取所有 person 和文件名
    if compare_person:
        persons = compare_person
    else:
        persons = []
        for k in combined_data.keys():
            if not len(combined_data[k].keys()) < len(file_list):
                persons.append(k)
        persons.sort()
        persons = [p for p in persons if p not in exlude]

    # 设置颜色
    colors = plt.cm.tab20.colors[:len(files)]  # 使用 matplotlib 内置调色板，确保颜色不同
    # print(colors)

    # 开始绘图
    fig, ax = plt.subplots(figsize=(12, 8))
    bar_width = 0.8 / len(files)  # 控制柱状体的宽度
    x_positions = range(len(persons))  # X 轴的位置

    # 绘制每个文件的柱状图
    for i, file in enumerate(files):
        # 获取当前文件对应的数据
        values = [combined_data[p][file] for p in persons]
        x_offsets = [x + i * bar_width for x in x_positions]  # 每个文件的柱体有不同的偏移

        # 绘制柱体
        bars = ax.bar(
            x_offsets,
            values,
            bar_width,
            label=file,
            color=colors[i]
        )

        # 在柱体上标注数值
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f'{height:.2f}',  # 保留两位小数
                ha='center',
                va='bottom',
                fontsize=10
            )

    # 设置图例和轴标签
    ax.set_title(f"Comparison of {compare_column} Across Files and Persons", fontsize=16)
    ax.set_xlabel("Person", fontsize=12)
    ax.set_ylabel(compare_column, fontsize=12)
    ax.set_xticks([x + (len(files) - 1) * bar_width / 2 for x in x_positions])
    ax.set_xticklabels(persons)
    ax.legend(title="File", fontsize=10, loc='lower right')
    plt.xticks(rotation=45)
    plt.tight_layout()

    # 保存图表
    output_path = f"{output_folder}/comparison_{compare_column}.png"
    plt.savefig(output_path)
    plt.close()
    print(f"图表已保存到: {output_path}")


if __name__ == '__main__':
    # file_dict = {
    #     "/root/sharedatas/mxg/fetigueDetectionTest/resluts/eegNet01/eegnet-results.xlsx": 'eegNet',
    #     "/root/sharedatas/mxg/fetigueDetectionTest/resluts/ESTCNN/ESTCNN-results.xlsx": 'ESTCNN',
    #     "/root/sharedatas/mxg/fetigueDetectionTest/resluts/CNNLSTM-500-256/moreEpoch/CNNLSTM-500-256-results.xlsx": 'CNNLSTM',
    #     "/root/sharedatas/mxg/fetigueDetectionTest/resluts/train-by-500-500-anti/moreEpoch/train-500-500-anti-results.xlsx": 'ours'
    # }

    # file_dict = {
    #     "/root/sharedatas/mxg/fetigueDetectionTest/resluts/eegNet/eegNet-results.xlsx": 'eegNet',
    #     "/root/sharedatas/mxg/fetigueDetectionTest/resluts/ESTCNN/ESTCNN-results.xlsx": 'ESTCNN',
    #     "/root/sharedatas/mxg/fetigueDetectionTest/resluts/CNNLSTM/CNNLSTM-results.xlsx": 'CNNLSTM',
    #     "/root/sharedatas/mxg/fetigueDetectionTest/resluts/train-by-500-500/forPerson/train-by-500-500-results.xlsx": 'ours'
    # }

    # file_dict = {
    #     "/root/sharedatas/mxg/fetigueDetectionTest/resluts/train-by-500-500-anti/forPerson/train-by-500-500-anti-results.xlsx": '500-500-a',
    #     "/root/sharedatas/mxg/fetigueDetectionTest/resluts/train-by-500-500/forPerson/train-by-500-500-results.xlsx": '500-500',
    # }

    # file_dict = {
    #     "/root/sharedatas/mxg/fetigueDetectionTest/resluts/eegNet/eegNet-results.xlsx": 'eegNet',
    #     "/root/sharedatas/mxg/fetigueDetectionTest/resluts/eegNet01/eegnet-results.xlsx": 'eegNet01',
    #     "/root/sharedatas/mxg/fetigueDetectionTest/resluts/ESTCNN/ESTCNN-results.xlsx": 'ESTCNN',
    #     "/root/sharedatas/mxg/fetigueDetectionTest/resluts/CNNLSTM-500-256/CNNLSTM-500-256-results.xlsx": 'CNNLSTM'
    # }

    # file_dict = {
    #     "/root/sharedatas/mxg/fetigueDetectionTest/resluts/CNNLSTM-500-256/kanTest-228/kcl-stn-results.xlsx": 'kcl-stn',
    #     "/root/sharedatas/mxg/fetigueDetectionTest/resluts/train-by-500-500/normal_test228/kct-stn-results.xlsx": 'kct-stn',
    #     "/root/sharedatas/mxg/fetigueDetectionTest/resluts/eegNet01/228/eegnet-results.xlsx": 'eegnet'
    # }

    # file_dict = {
    #     "/root/sharedatas/mxg/fetigueDetectionTest/resluts/train-by-500-500/normal_test228/kct-stn-results.xlsx": 'kct-stn',
    #     "/root/sharedatas/mxg/fetigueDetectionTest/resluts/train-by-500-500-anti/moreEpoch/train-500-500-anti-results.xlsx": '500-500-anti',
    #     "/root/sharedatas/mxg/fetigueDetectionTest/resluts/train-by-500-500/forPerson/train-by-500-500-results.xlsx": '500-500'
    #
    # }

    file_dict = {
        "/root/sharedatas/mxg/fetigueDetectionTest/resluts/train-by-500-500-anti/moreEpoch/train-500-500-anti-results.xlsx": '500-500-anti',
        "/root/sharedatas/mxg/fetigueDetectionTest/resluts/CNNLSTM-500-256/kanTest-228/kcl-stn-modify-results.xlsx": 'kcl-stn'
    }

    output_folder = "paint31-3"  # 替换为输出图表的文件夹路径
    os.makedirs(output_folder, exist_ok=True)  # 如果输出文件夹不存在，则创建
    compare_columns = ['train_result', 'valid_result', 'test_result']
    for c in compare_columns:
        compare_excel_files(file_dict, output_folder, compare_column=c)
