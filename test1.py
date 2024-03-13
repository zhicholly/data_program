'''
handle函数的最后一个参数是用来控制要不要加浮置板等其他信息的



'''




import pandas as pd
import seaborn as sns
from pylab import *
from scipy import signal
from scipy.fftpack import fft
import matplotlib.pyplot as plt
from test2 import chuli
import os

Fs = 0
data_S = 0
data_N = 0

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

pd.options.display.notebook_repr_html = False  # 表格显示
plt.rcParams['figure.dpi'] = 75  # 图形分辨率
sns.set_theme(style='darkgrid')  # 图形主题


def open_files(path, x):
    s_name = pd.read_csv(path)
    zs = None
    if x == 'noise':

        zs = mems_procAX(4, s_name['Noise'])
    elif x == 'ay':

        zs = mems_procAX(1, s_name['Ay'])
    elif x == 'az':

        zs = mems_procAX(3, s_name['Az'])
    return zs


def mems_procAX(channel, data):
    global Fs

    wz_n = []
    # s_t = time1[1]
    # data_S = round(s_t.timestamp() * 1000)
    # e_t = time1[len(time1) - 1]
    # data_N = round(e_t.timestamp() * 1000)
    # 横向2
    if channel == 1:
        Fs = 500  # round(len(data) / (int(data_N - data_S) / 1000))   #把毫秒换成秒，先算整段数据有多少秒，看每秒内的数据点数，在求整数
        avg = sum(data) / len(data)
        data = [float(ot - avg) for ot in data]
        b, a = butter_bandpass()
        # data转float以下计算
        # data = [float(ot) for ot in data]
        y = filter_matlab(b, a, data)
        fs = round(len(y) / (1 * Fs))  # 取整数 取5s
        # print('fs:', fs)

        for o in range(fs):
            data_s = y[o * Fs:(o + 5) * Fs]
            t, f, N = stationarity(data_s)
            # 将频率进行分组
            i1 = round(0.5 * N / Fs)
            i2 = round(5.4 * N / Fs)
            i3 = round(26 * N / Fs)
            sum1 = 0
            sum2 = 0
            sum3 = 0
            xm = 3.57
            # 分段计算平稳性指标
            for i in range(i1, i2):
                sum1 = sum1 + (xm * ((f[i] * 0.8 * t[i]) ** 0.1)) ** 10
            for i in range(i2 + 1, i3):
                sum2 = sum2 + (xm * (f[i] * 650 / (t[i] ** 3)) ** 0.1) ** 10
            for i in range(i3 + 1, N):
                sum3 = sum3 + (xm * (f[i] / t[i]) ** 0.1) ** 10
            wz = (1 * (sum1 + sum2 + sum3)) ** 0.1
            wz_n.append(round(wz, 4))
        # wz_n = [str(ot) for ot in wz_n]

        # 垂向3
    if channel == 3:
        Fs = 500
        b, a = butter_bandpass()
        # data转float以下计算
        avg = sum(data) / len(data)
        data = [float(ot - avg) for ot in data]
        y = filter_matlab(b, a, data)
        fs = round(len(y) / Fs)  # 取整数
        print('fs:',fs)

        for o in range(fs):
            data_s = y[o * Fs:(o + 5) * Fs]
            t, f, N = stationarity(data_s)
            # 将频率进行分组
            i1 = round(0.5 * N / Fs)
            i2 = round(5.9 * N / Fs)
            i3 = round(20 * N / Fs)
            sum1 = 0
            sum2 = 0
            sum3 = 0
            xm = 3.57
            # 分段计算平稳性指标
            for i in range(i1, i2):
                sum1 += (xm * ((f[i] * 0.325 * t[i]) ** 0.1)) ** 10
            for i in range(i2 + 1, i3):
                sum2 += (xm * ((f[i] * 400 / (t[i] ** 3)) ** 0.1)) ** 10
            for i in range(i3 + 1, N):
                sum3 += (xm * (f[i] / t[i]) ** 0.1) ** 10
            wz = (1 * (sum1 + sum2 + sum3)) ** 0.1
            wz_n.append(round(wz, 4))
        # wz_n = [str(ot) for ot in wz_n]

        # 噪声4
    if channel == 4:
        Fs = 1
        # fs = round(Fs / 10)
        num = round(len(data) / 500)
        data_num = []
        for i in range(num):
            data_z = data[i * 500:(i + 1) * 500]
            data_z = [float(ot) for ot in data_z]
            data_m = max(data_z)
            wz_n.append(data_m)
        # data_num = [str(ot) for ot in data_num]
    return wz_n


# def clean_data(choose,data):
#     data['ts'] = data['ts'].apply(lambda x: x[1:-1])
#     data['ts'] = pd.to_datetime(data['ts'], format="%Y-%m-%d %H:%M:%S")
#
#     if choose == 1:    #1的话就是选择加速度数据
#         data.dropna(subset=['ax'], how='any', inplace=True)
#         data['time_interval'] = data['ts'].diff()
#         data['time_interval'] = data['time_interval'].apply(lambda x: x.total_seconds())
#         data = data.reset_index(drop = True)
#     if choose == 2:  #2的话就是选择噪声维度
#         data.dropna(subset=['noise'], how='any', inplace=True)
#         data['time_interval'] = data['ts'].diff()
#         data['time_interval'] = data['time_interval'].apply(lambda x: x.total_seconds())
#         data = data.reset_index(drop=True)
#
#     return data


# 巴特沃斯滤波
def butter_bandpass():
    Fs = 500
    wp = 2 * 30 / Fs;
    ws = 2 * 60 / Fs;
    Rp = 1;
    As = 40;
    N, fn = signal.buttord(round(wp, 4), round(ws, 4), Rp, As)
    b, a = signal.butter(N, fn)
    return b, a


# 统计平均值和最大值的
def Consolidated_data(list_data):
    avg_data = []
    max_data = []
    for i in range(len(list_data)):
        avg_data.append(list_data[i]['ax'].avg())
        max_data.append(list_data[i]['ax'].max())
    return avg_dta, max_data


def filter_matlab(b, a, x):
    y = []
    NO1 = b[0] * x[1]
    y.append(NO1)
    for i in range(1, len(x)):
        y.append(0)
        for j in range(len(b)):
            if i >= j:
                y[i] = float(y[i]) + (b[j] * float(x[i - j]))
                j += 1
        for l in range(len(b) - 1):
            if i > l:
                y[i] = (y[i] - a[l + 1] * y[i - l - 1])
                l += 1
        i += 1
    return y


# fft、平稳性组成参数
def stationarity(y):
    N = len(y)
    fz = fft(y, N)

    abs_v = abs(fz)
    abs_v = [round(ot, 4) for ot in abs_v]
    abs_v = [ot * 2 / N for ot in abs_v]
    abs_v = [round(ot, 4) for ot in abs_v]
    t = []
    f = []
    for i in range(0, N):
        t_t = round((i) * Fs / N)
        t.append(t_t)
        f_f = abs_v[i] ** 3
        f.append(f_f)
    return t, f, N


# 只能是上行
def message_plus(distance):
    plus_data = pd.read_csv('D:/杭州/plus数据/20210422右线结果文件.csv')
    pplus_data = pd.read_csv('D:/杭州/plus数据/3号线曲线要素表2022.5月(1).csv')
    # 4种减振地段*2（曲直线）
    d = []
    r = []
    for dist in distance:
        for _, row_df1 in plus_data.iterrows():
            if row_df1['起始里程'] <= dist <= row_df1['终点里程']:
                if row_df1['减振地段'] == '一般':
                    if row_df1['曲线'] == '曲线段':
                        d.append('一般曲线段')
                        for _,k in pplus_data.iterrows():
                            if k['起始里程'] <= dist <= k['终点里程']:
                                r.append(k['曲线半径'])
                                break
                    if row_df1['曲线'] == '直线段':
                        d.append('一般直线段')
                        r.append(0)
                if row_df1['减振地段'] == '中等减振':
                    if row_df1['曲线'] == '曲线段':
                        d.append('中等减振曲线段')
                        for _,k in pplus_data.iterrows():
                            if k['起始里程'] <= dist <= k['终点里程']:
                                r.append(k['曲线半径'])
                                break
                    if row_df1['曲线'] == '直线段':
                        d.append('中等减振直线段')
                        r.append(0)
                if row_df1['减振地段'] == '高等减振':
                    if row_df1['曲线'] == '曲线段':
                        d.append('高等减振曲线段')
                        for _,k in pplus_data.iterrows():
                            if k['起始里程'] <= dist <= k['终点里程']:
                                r.append(k['曲线半径'])
                                break
                    if row_df1['曲线'] == '直线段':
                        d.append('高等减振直线段')
                        r.append(0)
                if row_df1['减振地段'] == '特殊减振':
                    if row_df1['曲线'] == '曲线段':
                        d.append('特殊减振曲线段')
                        for _,k in pplus_data.iterrows():
                            if k['起始里程'] <= dist <= k['终点里程']:
                                r.append(k['曲线半径'])
                                break
                    if row_df1['曲线'] == '直线段':
                        d.append('特殊减振直线段')
                        r.append(0)
                break

    return d,r

def find_midpoint(r):
    midpoints = []

    # 遍历数据找到值发生变化的位置
    k = 0
    for i in range(1, len(r)):
        if r[i] != r[k]:
            midpoint = (i - 1 + k) // 2  # 计算中间位置
            midpoints.append(midpoint)  # 将中间位置加入列表
            k = i

    midpoints.append((k + len(r)) // 2)
    return midpoints




# def plot_plus_data(plus_data):
#     for i in plus_data:
#         for j in i:
#             point = plt.scatter()


def handle(path, iterm, distance,op = 0):
    file_name = os.path.basename(path)
    # month = file_name[9:11]
    # day = file_name[11:13]
    s_name = file_name[:-4]
    df = pd.DataFrame()
    riqi = '车载平稳性数据'

    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False
    fig = plt.figure(figsize=(15, 7))
    ax = fig.add_subplot(111)
    distan = chuli(pd.read_csv(path), distance)

    plt.subplot(212)
    if iterm == 'noise':
        x = open_files(path, 'noise')
        df[riqi] = x
        distan_resampling = [abs(round(num, 0)) for num in
                             np.interp(np.linspace(0, len(distan), len(x)), np.arange(len(distan)), distan)]
        if lic1 > lic2:
            xkdistance = [lic1 - 75 - i for i in distan_resampling]
        else:
            xkdistance = [lic1 + 75 + i for i in distan_resampling]

        ####删除下面
        if op == 1:
            plus_data,r = message_plus(xkdistance)

            df['jiegou'] = plus_data
        df['里程'] = ['XK{}+{}'.format(int(i//1000),int(i-(i//1000)*1000)) for i in xkdistance]
        df['实际里程'] = [9358 - 75 - i for i in distan_resampling]
        # df.to_csv('D:/杭州/2024年1月17日监测图像/临时文件/{}1.csv'.format(s_name), index=False, encoding='utf_8_sig')

        plt.title('{}噪声数据'.format(s_name), fontsize=25)
        ax = sns.lineplot(data=df, x=df.index, y=riqi, label='噪声（dB)')
        ax = sns.lineplot([83 for i in range(len(x))], color='green', label="83")
        ax = sns.lineplot([85 for i in range(len(x))], color='yellow', label="85")
        plt.legend(loc='upper right', frameon=True, fancybox=True)
        plt.ylabel('噪声（dB）', fontsize=20)
        plt.xlabel('里程（M）', fontsize=20)
        plt.xticks(df.index[::3], distan_resampling[::3], rotation=90)
        # plt.tick_params( length=len(x)/2)

        max_index = x.index(max(x))
        plt.scatter(x=max_index, y=df[riqi][max_index], marker='o', color="r")
        plt.text(max_index, df[riqi][max_index], str(df[riqi][max_index]), ha='center', va='bottom')
        if xkdistance[1] > xkdistance[-1]:
            plt.xticks(df.index[::5],
                       ['XK{}+{}'.format(int(i // 1000), int(i - (i // 1000) * 1000)) for i in xkdistance[::5]],
                       rotation=90)
        else:
            plt.xticks(df.index[::5],
                       ['SK{}+{}'.format(int(i // 1000), int(i - (i // 1000) * 1000)) for i in xkdistance[::5]],
                       rotation=90)

    if iterm == 'ay':
        x = open_files(path, 'ay')
        df[riqi] = x
        # print(df)
        distan_resampling = [abs(round(num, 0)) for num in
                             np.interp(np.linspace(0, len(distan), len(x)), np.arange(len(distan)), distan)]
        if lic1 > lic2:
            xkdistance = [lic1 - 75 - i for i in distan_resampling]
        else:
            xkdistance = [lic1 + 75 + i for i in distan_resampling]


        if op == 1:
            plus_data,r = message_plus(xkdistance)

            df['jiegou'] = plus_data
        # df['里程'] = ['XK{}+{}'.format(int(i//1000),int(i-(i//1000)*1000)) for i in xkdistance]
        # df['实际里程'] = [9358 - 75 - i for i in distan_resampling]
        # df.to_csv('D:/杭州/2024年1月17日监测图像/临时文件/{}1.csv'.format(s_name), index=False, encoding='utf_8_sig')

        plt.title('{}横向加速度平稳性数据'.format(s_name), fontsize=25)
        sns.lineplot(data=df, x=df.index, y=riqi, label=riqi)
        sns.lineplot([2.5 for i in range(len(x))], color='green', label="2.5")
        sns.lineplot([2.75 for i in range(len(x))], color="y", label="2.75")
        sns.lineplot([3 for i in range(len(x))], color="r", label="3.0")
        plt.legend(loc='upper right', frameon=True, fancybox=True)
        plt.ylabel('', fontsize=20)
        plt.xlabel('里程（M）', fontsize=20)
        if xkdistance[1] > xkdistance[-1]:
            plt.xticks(df.index[::2],
                       ['XK{}+{}'.format(int(i // 1000), int(i - (i // 1000) * 1000)) for i in xkdistance[::2]],
                       rotation=90)
        else:
            plt.xticks(df.index[::2],
                       ['SK{}+{}'.format(int(i // 1000), int(i - (i // 1000) * 1000)) for i in xkdistance[::2]],
                       rotation=90)

        max_index = x.index(max(x))
        plt.scatter(x=max_index, y=df[riqi][max_index], marker='o', color="r")
        plt.text(max_index, df[riqi][max_index], str(df[riqi][max_index]), ha='center', va='bottom')

    if iterm == 'az':
        x = open_files(path, 'az')
        df[riqi] = x
        distan_resampling = [abs(round(num, 0)) for num in
                             np.interp(np.linspace(0, len(distan), len(x)), np.arange(len(distan)), distan)]
        if lic1 > lic2:
            xkdistance = [lic1 - 75 - i for i in distan_resampling]
        else:
            xkdistance = [lic1 + 75 + i for i in distan_resampling]

        if op == 1:
            plus_data,r = message_plus(xkdistance)

            df['jiegou'] = plus_data


        # ghj = pd.DataFrame()
        # ghj['noise'] = x
        # ghj['distance'] = distan_resampling
        # ghj.to_csv('D:/829项目/改造后.csv')
        plt.title('{}垂向加速度平稳性数据'.format(s_name), fontsize=25)
        sns.lineplot(data=df, x=df.index, y=riqi, label=riqi)
        sns.lineplot([2.5 for i in range(len(x))], color='green', label="2.5")

        sns.lineplot([2.75 for i in range(len(x))], color="y", label="2.75")
        sns.lineplot([3.0 for i in range(len(x))], color="r", label="3.0")

        plt.ylabel('', fontsize=20)
        plt.xlabel('里程', fontsize=13)
        plt.xlabel('里程（M）', fontsize=20)
        if xkdistance[1] > xkdistance[-1]:
            plt.xticks(df.index[::5],
                       ['XK{}+{}'.format(int(i // 1000), int(i - (i // 1000) * 1000)) for i in xkdistance[::5]],
                       rotation=45)
        else:
            plt.xticks(df.index[::5],
                       ['SK{}+{}'.format(int(i // 1000), int(i - (i // 1000) * 1000)) for i in xkdistance[::5]],
                       rotation=45)

        max_index = x.index(max(x))
        plt.scatter(x=max_index, y=df[riqi][max_index], marker='o', color="r")

        plt.text(max_index, df[riqi][max_index], str(df[riqi][max_index]), ha='center', va='bottom')



    if op == 1:
        #添加的外部数据
        kk = 78
        if iterm == 'noise':
            kk = 78
        if iterm == 'ay':
            kk = 1.5
        if iterm == 'az':
            kk = 1.5
        plt.scatter(x=[i for i in df[df['jiegou'] == '一般直线段'].index],
                    y=[kk] * len(df[df['jiegou'] == '一般直线段']),c = 'papayawhip',label = '一般直线段')
        plt.scatter(x=[i for i in df[df['jiegou'] == '一般曲线段'].index],
                    y=[kk] * len(df[df['jiegou'] == '一般曲线段']),c = 'gold',label = '一般曲线段')
        plt.scatter(x=[i for i in df[df['jiegou'] == '中等减振直线段'].index],
                    y=[kk] * len(df[df['jiegou'] == '中等减振直线段']), c='lime', label='中等减振直线段')
        plt.scatter(x=[i for i in df[df['jiegou'] == '中等减振曲线段'].index],
                    y=[kk] * len(df[df['jiegou'] == '中等减振曲线段']), c='seagreen', label='中等减振曲线段')
        plt.scatter(x=[i for i in df[df['jiegou'] == '高等减振直线段'].index],
                    y=[kk] * len(df[df['jiegou'] == '高等减振直线段']), c='violet', label='高等减振直线段')
        plt.scatter(x=[i for i in df[df['jiegou'] == '高等减振曲线段'].index],
                    y=[kk] * len(df[df['jiegou'] == '高等减振曲线段']), c='darkviolet', label='高等减振曲线段')
        plt.scatter(x=[i for i in df[df['jiegou'] == '特殊减振直线段'].index],
                    y=[kk] * len(df[df['jiegou'] == '特殊减振直线段']), c='lightcoral', label='特殊减振直线段')
        plt.scatter(x=[i for i in df[df['jiegou'] == '特殊减振曲线段'].index],
                    y=[kk] * len(df[df['jiegou'] == '特殊减振曲线段']), c='red', label='特殊减振曲线段')
        points = find_midpoint(r)
        for p in points:
            if int(r[p]) > 0:
                if iterm == 'noise':
                    plt.text(p,78.4,str(int(r[p])))
                if iterm == 'ay':
                    plt.text(p,1.5,str(int(r[p])))
                if iterm == 'az':
                    plt.text(p,1.5,str(int(r[p])))

    print(len(distan),len(x))
    plt.legend(loc='upper right', frameon=True, fancybox=True,fontsize = 6)
    # plt.savefig('D:/杭州/2024年1月17日监测图像/噪声/5/{}.jpg'.format(s_name),dpi=800, bbox_inches='tight')

    plt.show()
    # plt.close()


def get_distance(mems_name):
    global lic1, lic2
    dfA = pd.read_csv('D:/南京车载mems/真实里程/3号线.csv')
    file_name1 = mems_name
    parts = file_name1.split("-")
    part1 = parts[0]
    part2 = parts[1]
    biaoji1 = dfA[dfA['站点'] == part1]['标记'].tolist()[0]
    biaoji2 = dfA[dfA['站点'] == part2]['标记'].tolist()[0]
    if biaoji1 == biaoji2:
        lic1 = dfA[dfA['站点'] == part1]['里程'].tolist()[0]
        lic2 = dfA[dfA['站点'] == part2]['里程'].tolist()[0]
    else:  # 有些线路有分岔路口
        lic1 = dfA[dfA['站点'] == part1]['里程2'].tolist()[0]
        lic2 = dfA[dfA['站点'] == part2]['里程2'].tolist()[0]

    distan = abs(lic1 - lic2)

    return distan


if __name__ == "__main__":
    # path = "D:/5号线/6/上行/江川路-西渡站.csv"  # 20230203去火车站和  20230203往基隆方向
    # path2 = "D:/5号线/4/上行/莘庄站-春申路.csv"  # 20230203去火车站和  20230203往基隆方向
    # path3 = 'D:/5号线/6/下行/西渡站-江川路.csv'
    # path4 = 'D:/13号线/里程/金沙江路-大渡河路.csv'  # 浦三路-御桥路.csv
    path5 = 'D:\hangzhou/e45f01490217_20240313-130945.csv'
    distan = get_distance('柳洲东路-上元门')
    handle(path5,'ay',distan,op = 0)
    # path = 'D:/杭州/2024年1月17日拆分数据/2'
    # file_names = os.listdir(path)
    # for i in file_names:
    #     real = i[:-4]
    #     distan = get_distance(real)
    #     handle(path + '/' + i, 'ay', distan,op = 1)
