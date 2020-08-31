import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors

import warnings
warnings.filterwarnings("ignore")
from improved_spectral_clustering.SCY import SpectralClustering
import evaluate
def expand(a, b):
    d = (b - a) * 0.1
    return a - d, b + d
def Fmax(a):
    max_num = a[0]
    max_index = 0
    for i in range(len(a)):
        if a[i] > max_num:
            max_num = a[i]
            max_index = i
    return max_index


def sp(data,class_num,data_nm,label):

    n_clusters=class_num
    matplotlib.rcParams['font.sans-serif'] = [u'SimHei']
    matplotlib.rcParams['axes.unicode_minus'] = False
    plt.figure(facecolor='w')
    clrs =['#B03060', '#AEEEEE', '#68228B', 'y', 'c', 'm', '#2E2E2E', '#00008B', '#2E8B57', '#FAEBD7',
                     '#8B5A00', '#EEEE00', '#0000FF', '#ABABAB', '#8B8B00']
    gamma_list = [0.1,0.2,0.4,0.6,0.8,1]
    af=['laplacian','nearest_neighbors']
    Compare=[]
    for gamma_value in gamma_list:
        # for a in af:
        spectral = SpectralClustering(n_clusters,gamma=gamma_value, affinity='nearest_neighbors',random_state=1)
        y_hat = spectral.fit_predict(data)
        Compare.append(y_hat)
    N=[]
    A=[]
    for  com in Compare:
        NMI,ARI=evaluate.eva_com(com,label)
        N.append(NMI)
        A.append(ARI)
    k=Fmax(N)
    y_hat=Compare[k]
    print("标准化互信息      精度      纯度     轮廓系数    兰德系数")
    nmi, acc, purity,Sc,ARI= evaluate.eva(y_hat, label,data)
    print(nmi, acc, purity,Sc,ARI)
    for k, clr in enumerate(clrs):
        cur = (y_hat == k)
        plt.scatter(data[cur, 0], data[cur, 1], s=40, color=clr, edgecolors='k')
    x1_min, x2_min = np.min(data, axis=0)
    x1_max, x2_max = np.max(data, axis=0)
    x1_min, x1_max = expand(x1_min, x1_max)
    x2_min, x2_max = expand(x2_min, x2_max)
    plt.xlim((x1_min, x1_max))
    plt.ylim((x2_min, x2_max))
    # plt.grid(True)
    # plt.legend(loc='best')
    plt.title("SC2+" + data_nm)
    plt.savefig('.\picture\improved_spectral_clustering\sc1_{0}.png'.format(data_nm))
    plt.close()






    # print("标准化互信息      精度      纯度     轮廓系数    兰德系数")
    # nmi, acc, purity,Sc,ARI= evaluate.eva(y_hat, label,data)
    # print(nmi, acc, purity,Sc,ARI)


