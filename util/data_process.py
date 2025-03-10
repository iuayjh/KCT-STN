import numpy as np
from tqdm import tqdm

data_source = '../tmp_data/data-samples-500-500.npy'

if __name__ == '__main__':
    print('start')
    data = np.load(data_source)
    # 采样比例
    sampling_ratios = np.arange(0.1, 1.0, 0.1)  # 从 10% 到 90%，步长为 10%

    # 设置随机种子（可选）
    np.random.seed(42)

    # 输出文件路径模板
    output_template = "../tmp_data/data-samples-500-500_{:.0f}%.npy"

    # 按比例随机采样并保存
    for ratio in sampling_ratios:
        num_samples = int(data.shape[0] * ratio)  # 计算采样大小
        sampled_indices = np.random.choice(data.shape[0], num_samples, replace=False)  # 随机采样索引
        sampled_data = data[sampled_indices]  # 根据索引提取数据
        output_file = output_template.format(ratio * 100)  # 格式化输出文件名
        np.save(output_file, sampled_data)  # 保存为 .npy 文件
        print(f"保存 {ratio * 100:.0f}% 采样数据到: {output_file}")