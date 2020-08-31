# Hierarchical Clustering
import matplotlib.pyplot as plt
from itertools import cycle, islice
import seaborn as sns
import os
import numpy
# Importing the libraries
import numpy as np
import evaluate
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
def HC(data, class_num, data_nm,label):
    k = class_num
    # Z = sch.linkage(data, method='average',metric='euclidean')
    # sch.dendrogram(Z)
    # plt.savefig('.\picture\hierarchical_clustering\c_{0}.png'.format(data_nm))
    # plt.close()
    # plt.figure()
    # sns.clustermap(data,method='average',metric='euclidean',cmap='RdYlBu_r')
    # plt.savefig('.\picture\hierarchical_clustering\c1_{0}.png'.format(data_nm))
    # plt.close()


    hc = AgglomerativeClustering(k, affinity = 'euclidean', linkage = 'ward')
    y_hc = hc.fit_predict(data)
    # print(len(y_hc))
    # print(len(label.reshape(-1)))
    print("标准化互信息      精度      纯度     轮廓系数    兰德系数")
    nmi, acc, purity,Sc,ARI = evaluate.eva(y_hc, label,data)
    print(nmi, acc, purity,Sc,ARI)
    colors = ['#B03060', '#AEEEEE', '#68228B', 'y', 'c', 'm','#2E2E2E','#00008B','#2E8B57','#8B0000','#8B5A00','#EEEE00','#CDCDB4','#ABABAB','#8B8B00']
    plt.figure()
    for i in range(k):
        for j in range(0,6):
            color = colors[i % len(colors)]
            plt.scatter(data[y_hc == i, 0], data[y_hc == i, 1], s = 6, c = color)
    plt.title("AgglomerativeClustering+" + data_nm)
    # plt.legend(loc='best')
    plt.savefig('.\picture\hierarchical_clustering\hc_{0}.png'.format(data_nm))
    plt.close()

