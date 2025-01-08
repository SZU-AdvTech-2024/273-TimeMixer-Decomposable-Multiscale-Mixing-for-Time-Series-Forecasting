import pandas as pd
import h5py
import numpy as np


h5_file = '../dataset/SST/201101.h5'
csv_file = '../dataset/SST/201101.csv'


"""
1、首先检查文件中存储的数据集名称和结构（数据维度和数据类型）
2、然后读取数据集并转换为NumPy数组
3、根据数据的维度，转换为 Pandas DataFrame
4、使用Pandas将DataFrame保存为csv文件
"""
with h5py.File(h5_file, 'r') as f:
    # 查看所有键
    print("Keys in the H5 file: ", list(f.keys()))
    # 查看某个键的数据集形状和属性
    for key in list(f.keys()):
        print(f"{key} dataset shape: {f[key].shape}")
        print(f"{key} data type: {f[key].dtype}")

    dataset_name = 'sst' # 名称一定要替换为实际数据集的名称
    data = f[dataset_name][:] # 读取数据为NumPy数组
    # print(f"加载数据形状：{data.shape}")
    # print(f"数据类型：{data.dtype}")

# 确保数据是二维数组，方便转换为DataFrame
if len(data.shape) == 2:
    df = pd.DataFrame(data)
elif len(data.shape) == 3:
    # 如果是三维数据，展平成二维表格
    df = pd.DataFrame(data.reshape(data.shape[0], -1))
else:
    raise ValueError("data has unsupported dimensions")

# 查看数据前几行
# print(df.values)


# 如果数据量很大，生成的csv文件就会很大，可以使用压缩格式或者分批保存
# df.to_csv(csv_file, index=False, compression='gzip')
# df.to_csv(csv_file, index=False) # 保存为CSV文件，不包括索引列
# print(f"数据保存至{csv_file}")

# 文件过大时的优化：逐块读取和保存
# with h5py.File(h5_file, 'r') as f:
#     dataset = f[dataset_name]
#     chunk_size = 1000 # 每次处理 1000 行
#     for i in range(0, dataset.shape[0], chunk_size):
#         chunk_data = dataset[i:i+chunk_size,:]
#         chunk_df = pd.DataFrame(chunk_data)
#         # 保存到 csv（追加模式）
#         chunk_df.to_csv(csv_file, index=False, mode='a', header=(i==0))
