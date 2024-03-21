import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt


def butterworth_filtery(data, order):
    fs = 1 / 0.00616

    cutoff = 50
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='lowpass', analog=False)
    y = filtfilt(b, a, data)
    return y


# 只要中间切分的点，不要起始和终点
def split_data2(path, number):
    # number 代表着几个区间
    df = pd.read_csv(path)
    df['Ax'] = pd.to_numeric(df['Ax'], errors='coerce')

    # 删除含有NaN值的行，即删除包含字符串的行
    df.dropna(subset=['Ax'], inplace=True)
    var_mean = df['Ay'].groupby(df.index // 10000).var()
    print(var_mean)
    sorted_indices = sorted(range(len(var_mean)), key=lambda i: var_mean[i])

    smallest_values = sorted(sorted_indices[:number], reverse=True)  # 找出方差最小的区间
    result = []
    for i in range(len(smallest_values)):
        if i == 0 or smallest_values[i] - smallest_values[i - 1] != -1:
            result.append(smallest_values[i])
    result =result[::-1]

    print(result)
    print(len(result))
    dict_index = {}

    dict_index[0] = df.iloc[: result[0] * 10000]

    if len(result) < 2:
        dict_index[len(result)] = df.iloc[result[0] * 10000:]
    else:
        for i in range(1, len(result)):
            dict_index[i] = df.iloc[
                            result[i - 1] * 10000: result[i] * 10000
                            ]
        dict_index[len(result)] = df.iloc[result[-1] * 10000:]

    return dict_index, result


kk, values = split_data2('D:/南京车载mems/原始数据/S1号线/e45f01490217_20240321-144908.csv',16)
# # #
# # # #

name = pd.read_csv('D:/南京车载mems/真实里程/S1机场线.csv')['站点'].tolist()
name = name[::-1]
print(name)
print(len(name))

# 上面的是没有支线的用法，下面是有支线的，判断下支线
# name_df = pd.read_csv('D:/829项目/真实里程/5号线（主支线）.csv')
# zhuxian = name_df[name_df['支线主线'] != 2].reset_index()
# name = zhuxian['站点'].tolist()[6:]
# print(name)

for i in range(1,len(name)):

    kk[i-1].to_csv('D:/南京车载mems/拆分数据/S1机场线/空港新城江宁-南京南站/{}-{}.csv'.format(name[i-1],name[i]),index = False)
    print(name[i - 1], name[i], i)
print(name)

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

df = pd.read_csv('D:/南京车载mems/原始数据/S1号线/e45f01490217_20240321-144908.csv')
df['Ax'] = pd.to_numeric(df['Ax'], errors='coerce')  # 传感器极少数时候会有非数字，i dont know why

# 删除含有NaN值的行，即删除包含字符串的行
df.dropna(subset=['Ax'], inplace=True)
plt.plot(df['Ax'], alpha=0.5)
plt.scatter(x=[i * 10000 for i in values], y=[df['Ax'][i] for i in [i * 10000 for i in values]], marker='o', color="r")
for i in range(len(values)):
    plt.text(values[i] * 10000, df['Ax'][values[i] * 10000], name[i+1 ], ha='center', va='bottom', fontsize=10)
plt.show()
