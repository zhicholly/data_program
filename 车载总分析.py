import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.signal import butter, filtfilt,medfilt
from scipy.fftpack import fft
from scipy import signal
import datetime
import seaborn as sns
from pylab import *
from mpl_toolkits.mplot3d import Axes3D

from test1 import mems_procAX
from test2 import chuli

def handle (path, fangxiang=None,low_bound1 = None):
    up_title = os.path.basename(path)#f返回最后一个/后面的名
    file_names = os.listdir(path)

    zongde = pd.DataFrame()
    for i in file_names:
        try:
            print('处理{}文件'.format(i))
            namedic = []
            locale = []
            quancheng_local = []
            distance1 = []
            kk = pd.DataFrame()
            num_y = 0
            name = i[:-4]
            label1 = []
            label2 = []
            # print(name)
            real_distan,lic1,lic2 ,biaoji1,biaoji2 = get_distance(name)
            df = pd.read_csv(path + '/' + i)

            if fangxiang  == '垂向':
                zs = mems_procAX(3,df['Az'])
            elif fangxiang == '横向':
                zs = mems_procAX(1, df['Ay'])
            elif fangxiang == '噪声':
                zs = mems_procAX(4,df['Noise'])




            calculate_distan = chuli(df,real_distan)

            distan_resampling = [abs(round(num, 2)) for num in
                                 np.interp(np.linspace(0, len(calculate_distan), len(zs)), np.arange(len(calculate_distan)), calculate_distan)]

            kk['distance'] = distan_resampling
            kk['zs'] = zs
            # kk.to_csv('D:/829项目/下行横向2级超限里程1.csv',encoding='utf_8_sig',index = False)
            dic,values = calculate_durations2(zs,low_bound1,200,distan_resampling)            # zs时噪声，平稳性（噪声）低限值，平稳性（噪声）的高限值，距离
            # print(dic,values)
            for i in dic:
                namedic.append(name)
                label1.append(biaoji1)
                label2.append(biaoji2)


                for j in range(len(i)):
                    if j == 0:
                        locale.append(i[0])
                        if lic1 < lic2:
                            quancheng_local.append(lic1+i[0])
                        else:
                            quancheng_local.append(lic1-i[0])
                    else:
                        distance1.append(i[1])
            data = {'区间': namedic, '里程': locale, '平稳性指标': values, '全程里程': quancheng_local, '长度': distance1,
                    '标记1': label1, '标记2': label2}
            kk = pd.DataFrame(data)
            if zongde.empty:
                zongde = kk
            else:
                zongde = pd.concat([zongde, kk], ignore_index=True)

        except Exception as e:
            print(f"在处理文件夹 {i} 时出现错误：{e}")
        continue




    added_data = add_data(zongde)

    added_data.to_csv('D:/926大渡河金沙江/横向2.5/{}/{}{}平稳性超限里程.csv'.format(xianlu,up_down,fangxiang),encoding='utf_8_sig',index = False)


def add_data(data):
    global xianlu, up_down


    df = pd.read_csv('D:/829项目/清洗过的数据.csv')
    filter_df = df[(df['行别/股道'] == up_down) & (df['线名'] == xianlu)]

    quzhi = []
    r = []
    length = []
    for index, row_df1 in data.iterrows():
        if row_df1['标记1'] == row_df1['标记2']:
            lic = float(row_df1['全程里程'])
            matching_rows = filter_df[(filter_df['起点标记'] == row_df1['标记1']) & (filter_df['起点数字'] < lic) & (
                        filter_df['终点数字'] > lic)]
            if matching_rows.empty:
                matching_rows = filter_df[(filter_df['起点数字'] > lic) & (filter_df['终点数字'] < lic)]

            # print('matching_rows:',matching_rows)
            if matching_rows.empty:
                quzhi.append('直线')
                r.append(0)
                length.append(0)
            else:
                r.append(matching_rows['曲线半径（m）'].values[0])
                length.append(matching_rows['曲线长度（m）'].values[0])
                quzhi.append('曲线')
        else:
            quzhi.append('不判断')
            r.append(0)
            length.append(0)

    data['曲直线'] = quzhi
    data['曲线半径'] = r
    data['曲线长度'] = length
    return data



def get_distance(mems_name):
    real_d = 0
    for file in os.listdir('D:/829项目/真实里程'):
        if xianlu in file:
            dfA = pd.read_csv('D:/829项目/真实里程/{}'.format(file))
    file_name1 = mems_name
    parts = file_name1.split("-")
    part1 = parts[0]
    part2 = parts[1]

    biaoji1 = dfA[dfA['站点'] == part1]['标记'].tolist()[0]
    biaoji2 = dfA[dfA['站点'] == part2]['标记'].tolist()[0]
    if biaoji1 == biaoji2:
        lic1 = dfA[dfA['站点'] == part1]['里程'].tolist()[0]
        lic2 = dfA[dfA['站点'] == part2]['里程'].tolist()[0]
    else:
        lic1 = dfA[dfA['站点'] == part1]['里程2'].tolist()[0]
        lic2 = dfA[dfA['站点'] == part2]['里程2'].tolist()[0]

    distan = abs(lic1 - lic2)
    # if lic1 < lic2:
    #     licheng = lic1+distan
    # else :
    #     licheng = lic2-distan
    return distan,lic1,lic2,biaoji1,biaoji2








#计算区间长度的函数
def calculate_durations(returns, lower_bound, upper_bound,distance):
    durations = []
    start = None

    for i, ret in enumerate(returns):
        if lower_bound <= float(ret) <= upper_bound:
            if start is None:
                start = i
        elif start is not None:
            # if start != i - 1:
            durations.append((distance[start], round(distance[i-1]-distance[start],2)))
            start = None

    if start is not None:

        durations.append((distance[start], round(distance[len(returns) - 1]-distance[start],2)))

    return durations



def calculate_durations2(returns, lower_bound, upper_bound, distance):
    # global up_down
    durations = []
    value = 0
    values= []
    start = None

    for i, ret in enumerate(returns):
        if lower_bound <= float(ret) <= upper_bound:
            if start is None:
                start = i
                value = ret
            else:
                if ret > value:
                    value = ret  #  这个就是取这个区间平稳性最大值
        elif start is not None:    #如果ret超出这个范围，并且start不是空的
            if up_down == '上行':
                durations.append((distance[start]+75, round(distance[i+1] - distance[start-1]+50, 2)))   #上行的区间的里程加了75m，并且距离还是前后各增加了1s
            else:
                durations.append((distance[start]-75, round(distance[i+1] - distance[start-1]+50, 2)))   #下行的区间减了75m

            values.append(value)
            start = None

    if start is not None:

        durations.append((distance[start], round(distance[len(returns) - 1] - distance[start], 2)))# 这个基本用不到，用到的，可以考虑去除？
        values.append(value)
    return durations,values





if __name__ == "__main__":

    root_folder = 'D:\829项目'   #项目地址
    keyword = "拆分后数据"
    matching_subfolders = []
    star= 0   #这个计数用的
    for folder_name, subfolders, filenames in os.walk(root_folder):
        if keyword in folder_name:
            matching_subfolders.extend([os.path.join(folder_name, subfolder) for subfolder in subfolders])
    for i in matching_subfolders:
        first_slash_index = i.find("\\")
        star+= 1
        # 找到斜杠的位置
        second_slash_index = i.find("\\", first_slash_index + 1)
        third_slash_index = i.find("\\", second_slash_index + 1)
        four_slash_index = i.find("\\", third_slash_index + 1)
        # 提取两个斜杠之间的字符串
        if third_slash_index != -1 and four_slash_index != -1:
            xianlu = i[second_slash_index + 1:third_slash_index]
            print(xianlu,)
            up_down = i[four_slash_index + 1:four_slash_index + 3]
        print('{}now 开始执行{}中的文件'.format(star,i))

        handle(i,'横向',low_bound1= 2.5)