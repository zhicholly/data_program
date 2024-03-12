import math

import numpy as np
import matplotlib.pyplot as plt
import pwlf
import pandas as pd
import os
from scipy.signal import butter, filtfilt,medfilt
from scipy.fftpack import fft
from scipy import signal

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


def open_file(path):
    file_data= pd.read_csv(path)

    chuli2(file_data,2386)

from scipy.signal import resample



def clean_data(data):
    data['ts'] = data['ts'].apply(lambda x: x[1:-1])
    data['ts'] = pd.to_datetime(data['ts'], format="%Y-%m-%d %H:%M:%S")
    data.dropna(subset=['ax'], how='any', inplace=True)
    data['time_interval'] = data['ts'].diff()
    data['time_interval'] = data['time_interval'].apply(lambda x: x.total_seconds())
    data = data.reset_index(drop = True)
    return data

    #少了一步滤波
def chuli(df,licheng):
    data = df
    filter_x =butterworth_filtery(data['Ax'],1)
    speed_x = calculate_AYspeeds(filter_x)
    speed_x1 = calculate_AY1speeds(filter_x)
    xx = Data_fusion(speed_x,speed_x1)
    # print(yy[39000], yy[40000], yy[40750])
    distan = distance_rong(xx,licheng)
    # print(distan[39000],distan[40000],distan[40750],distan[27000],distan[19500])
    plt.subplot(211)
    plt.title('speed',fontsize = 20)
    plt.plot(speed_x)

    return distan


def chuli2(df,licheng):
    data = df
    filter_x =butterworth_filtery(data['Ax'],1)
    speed_x = calculate_AYspeeds(filter_x)
    speed_x1 = calculate_AY1speeds(filter_x)
    xx = Data_fusion(speed_x,speed_x1)
    # xx = cal_distance(speed_x)
    # print(yy[39000], yy[40000], yy[40750])
    distan = distance_rong(speed_x,licheng)
    # print(distan[39000],distan[40000],distan[40750],distan[27000],distan[19500])
    # plt.plot(data['Gz'].rolling(550).mean())
    # plt.title('陀螺仪数据')
    plt.plot(data['Gx'].rolling(550).mean(),label = 'Gx')
    plt.plot(data['Gy'].rolling(550).mean(),label = 'Gy')
    plt.plot(data['Gz'].rolling(550).mean(),label = 'Gz')
    # plt.plot(speed_x)
    xkdistance = [round(7289  + i, 0) for i in distan[::500]]
    jj = ['XK{}+{}'.format(int(i // 1000), int(i - (i // 1000) * 1000)) for i in xkdistance]
    plt.xticks(data.index[::500],jj,rotation = 90)
    # plot_x(distan)
    plt.legend()
    plt.show()



#滤波函数
#scipy.signal.butter 函数返回两个数组，分别为滤波器的系数：一个表示滤波器的输入部分，另一个表示滤波器的输出部分。
# 截止频率为 cutoff，并以采样率 fs 和滤波器阶数 order 应用。
# def butter_bandpass():
#     Fs = 1/0.00616
#     wp = 2 * 30 / Fs;
#     ws = 2 * 60 / Fs;
#     Rp = 1;
#     As = 40;
#     N, fn = signal.buttord(round(wp, 4), round(ws, 4), Rp, As)
#     b, a = signal.butter(N, fn)
#     return b, a
#
#
# def filter_matlab(b, a, x):
#     y = []
#     NO1 = b[0] * x[1]
#     y.append(NO1)
#     for i in range(1, len(x)):
#         y.append(0)
#         for j in range(len(b)):
#             if i >= j:
#                 y[i] = float(y[i]) + (b[j] * float(x[i - j]))
#                 j += 1
#         for l in range(len(b) - 1):
#             if i > l:
#                 y[i] = (y[i] - a[l + 1] * y[i - l - 1])
#                 l += 1
#         i += 1
#     return y



def butterworth_filtery(data, order):
     fs = 500

     cutoff = 50
     nyq = 0.5 * fs
     normal_cutoff = cutoff / nyq
     b, a = butter(order, normal_cutoff, btype='lowpass', analog=False)
     y = filtfilt(b, a, data)
     return y


def med_filter(data):
    window_size = 499
    filtered_data = medfilt(data, kernel_size=window_size)
    return filtered_data

def Data_fusion(speed1,speed2):
    weight =0
    speed =0
    speeds =[]
    for i in range(len(speed1)):
        weight += 1/len(speed1)
        speed = speed1[i]*(1-weight)+speed2[i]*weight
        speeds.append(speed)
    # if sum(speeds) < 0:
    #     speeds = [abs(i) for i in speeds]
    return speeds










def power_spectral_density(filter_ax,measure_ax):
    fs =500
    f,p1 = signal.periodogram(filter_ax,fs)
    k,p2 = signal.periodogram(measure_ax,fs)
    return np.array(p1)/np.array(p2)


def plot_x_y (x,y,z):
    plt.figure()

    plt.plot(x)
    plt.plot(y)
    plt.plot(z)
    plt.show()
def plot_x(x):
    plt.figure()
    plt.plot(x)
    plt.show()

def plot_distances(x,y,z):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x, y, z)
    ax.set_title("Motion Plot")
    ax.set_xlabel("X Axis")
    ax.set_ylabel("Y Axis")
    ax.set_zlabel("Z Axis")
    ax.axis('equal')
    plt.show()

def calculate_AYspeeds(ax):
    speed = 0
    speeds = []
    avg = []

    environment = np.mean(ax[:500])
    error = np.mean(ax-environment)
    for i in range(50,len(ax)-10):

        ax_avg = ((ax[i]+ax[i-1])/2)-environment-error      #一个时间段的平均加速度
        avg.append(ax_avg)
        speed += ax_avg*(1/500)  #平均速度
        speeds.append(speed)
    return speeds

def distance_rong(speed,licheng):
    speed = np.array(speed)
    s = np.cumsum(licheng * speed / np.sum(speed))
    return list(s)





def calculate_AY1speeds(ax):

    ax = ax[::-1]
    ax = [-x for x in ax]
    speed = 0
    speeds = []
    avg = []
    environment = np.mean(ax[:500])
    error = np.mean(ax-environment)
    for i in range(50,len(ax)-10):

        ax_avg = ((ax[i]+ax[i-1])/2)-environment#-error      #一个时间段的平均加速度
        avg.append(ax_avg)
        speed += ax_avg*(1/500)   #平均速度
        speeds.append(speed)
    speeds = speeds[::-1]
    return speeds


def cal_distance(speeds):
    distance = 0
    distances = []
    for i in range(1,len(speeds)):
        avg = (speeds[i-1]+speeds[i])/2
        distance += avg*0.002
        distances.append(distance)
    return distances



#徐汇交大913.874米   交大徐汇907.844米
#交大江苏2128.641米  江苏交大2132.31米
#江苏隆德1361.077米  隆德江苏1355.983米
#隆德曹杨999.406米   曹杨隆德1001.208米
#曹杨枫桥725.726米   枫桥曹杨721.591米
#


if __name__ == "__main__":
    path = 'E:/mems_20230222_152058-152105.csv'
    path10 = 'D:/暂存数据/罗山.csv'
    path100 = "D:/南京4号线/南京地铁4号线徐庄-汇通路添乘数据/mems_20230213_163440-163816金马路-徐庄.csv"
    path22 = 'D:/13号线/里程/大渡河路-金沙江路.csv'  #3660
    path222 = 'D:/上海11号线/20230214上海11号线全线动态/迪士尼-花桥-迪士尼/mems_20230214_160958-161254秀沿路-康新公路.csv'
    path4 = 'D:/杭州/2024年1月17日拆分数据/2/文一西路-绿汀路站.csv'
    open_file(path4)
