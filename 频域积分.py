import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter

data = pd.read_csv('D:/管片加速度积分/振动位移/宁波/962/download (39)/d83add6120a2VK_20240119-111240.csv').iloc[:,0]
# plt.subplot(311)
# plt.plot(data)





def jifen(data,sf):
    f_min = 0.5
    f_max = 5
    Data = np.fft.fft(data)
    N = len(Data)
    f = np.fft.fftfreq(N, 1 / sf)
    mask = (f >= f_min) & (f <= f_max)
    A_integral = np.zeros_like(Data, dtype=np.complex128)
    B_integral = np.zeros_like(Data, dtype=np.complex128)

    A_integral[mask] = Data[mask] / (2j * np.pi * f[mask])
    # plt.plot(A_integral)
    # plt.show()
    B_integral[mask] = A_integral[mask] / (2j * np.pi * f[mask])
    print(B_integral[mask].shape)
    plt.title('this is B_INTEGRAL')
    plt.plot(B_integral.real,label = 'acc',alpha = 0.4)
    plt.show()
    weiyi = np.fft.ifft(B_integral[:40000]).real
    sudu = np.fft.ifft(A_integral[:40000]).real
    return weiyi,sudu

weiyi,sudu = jifen(data,2000)
plt.subplot(312)
plt.plot(weiyi)
# plt.show()





def bandpass_filter(signal, lowcut, highcut, fs, order=4):
    """
    带通滤波器函数

    参数:
    signal : 输入信号
    lowcut : 低截止频率
    highcut : 高截止频率
    fs : 采样频率
    order : 滤波器的阶数（默认为4）

    返回:
    filtered_signal : 经过带通滤波后的信号
    """
    # 归一化截止频率
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq

    # 设计带通滤波器
    b, a = butter(order, [low, high], btype='band')

    # 应用滤波器
    filtered_signal = lfilter(b, a, signal)


    return filtered_signal

# plt.subplot(313)
# banddata = bandpass_filter(weiyi,0.6,60,2000)

# plt.plot(jifen(banddata,2000)*10)
# plt.show()


real_weiyi =(pd.read_csv('D:/管片加速度积分/振动位移/宁波/962/download (40)/d83add6120a2_20240119-111241.csv').iloc[:,0]-211.475)*-0.001
fft_weiyi = np.fft.fft(real_weiyi)
ifft_weiyi = np.fft.ifft(fft_weiyi)
plt.plot(ifft_weiyi,alpha = 0.4)

plt.legend()
plt.show()

sudu2 = pd.Series(sudu).rolling(10).mean()
velocity_data = np.cumsum(sudu2)

# 将速度数据乘以采样时间间隔，得到位移
displacement_data = velocity_data * (1 / 2000)
plt.plot(displacement_data)
# plt.show()
