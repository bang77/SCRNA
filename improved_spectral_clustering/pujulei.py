import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
import warnings
warnings.filterwarnings("ignore")
from sklearn.cluster import spectral_clustering
from sklearn.metrics import euclidean_distances
import evaluate
def expand(a, b):
    d = (b - a) * 0.1
    return a - d, b + d
def sp(data,class_num,data_nm,label):
    n_clusters=class_num
    matplotlib.rcParams['font.sans-serif'] = [u'SimHei']
    matplotlib.rcParams['axes.unicode_minus'] = False
    m = euclidean_distances(data, squared=True)
    # print(m)
    sigma = np.median(m)
    plt.figure(figsize=(12, 8), facecolor='w')
    plt.suptitle(u'谱聚类', fontsize=20)
    clrs =['#B03060', '#AEEEEE', '#68228B', 'y', 'c', 'm', '#2E2E2E', '#00008B', '#2E8B57', '#FAEBD7',
                     '#8B5A00', '#EEEE00', '#0000FF', '#ABABAB', '#8B8B00']
    # print(len(clrs))

    assess=[]
    for i, s in enumerate(np.logspace(-2, 0, 6)):

        af = np.exp(-m ** 2 / (s ** 2)) + 1e-6
        y_hat = spectral_clustering(af, n_clusters=n_clusters, assign_labels='kmeans', random_state=1)
        # assess.append(y_hat)
        plt.subplot(2, 3, i + 1)
        for k, clr in enumerate(clrs):
            cur = (y_hat == k)
            plt.scatter(data[cur, 0], data[cur, 1], s=40, color=clr, edgecolors='k')
        x1_min, x2_min = np.min(data, axis=0)
        x1_max, x2_max = np.max(data, axis=0)
        x1_min, x1_max = expand(x1_min, x1_max)
        x2_min, x2_max = expand(x2_min, x2_max)
        plt.xlim((x1_min, x1_max))
        plt.ylim((x2_min, x2_max))
        plt.grid(True)
        plt.title(u'sigma = %.2f' % s, fontsize=16)
    # print(y_hat)
    print("标准化互信息      精度      纯度     轮廓系数    兰德系数")
    nmi, acc, purity,Sc,ARI= evaluate.eva(y_hat, label,data)
    print(nmi, acc, purity,Sc,ARI)
    plt.tight_layout()
    plt.title("SC1+" + data_nm)
    plt.subplots_adjust(top=0.9)
    plt.savefig('.\picture\improved_spectral_clustering\sc1_{0}.png'.format(data_nm))
    plt.close()

