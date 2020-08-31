from sklearn.cluster import DBSCAN
from sklearn import metrics
import evaluate
import numpy
import seaborn
from itertools import cycle, islice
import matplotlib.pyplot as plt
def dbscan(data,data_nm,class_num,label):
    # 设置半径为eps，最小样本量为min_samples，建模
    db = DBSCAN(eps=0.82, min_samples=2).fit(data)
    labels = db.labels_
    # print(len(labels))
    # print(len(label.reshape(-1)))
    print("标准化互信息      精度      纯度     轮廓系数    兰德系数")
    nmi, acc, purity,SC,ARI= evaluate.eva(labels, label,data)
    print(nmi, acc, purity,SC,ARI)
    # 计算噪声点个数占总数的比例
    # raito = len(labels[labels[:] == -1]) / len(labels)
    # print('噪声比:', format(raito, '.2%'))
    plotRes(data, labels, class_num, data_nm)
def plotRes(data, clusterRes, clusterNum, data_nm):
    nPoints = len(data)
    scatterColors = ['#B03060', '#AEEEEE', '#68228B', 'y', 'c', 'm', '#2E2E2E', '#00008B', '#2E8B57', '#FAEBD7',
                     '#8B5A00', '#EEEE00', '#0000FF', '#ABABAB', '#8B8B00']
    plt.figure()
    for i in range(clusterNum):
        color = scatterColors[i % len(scatterColors)]
        x1 = []
        y1 = []
        for j in range(nPoints):
            if clusterRes[j] == i:
                x1.append(data[j, 0])
                y1.append(data[j, 1])
        plt.scatter(x1, y1, c=color, s=40, edgecolors='k')
    # plt.legend(loc='best')
    plt.title("dbscan+" + data_nm)
    plt.savefig('.\picture\dbscan\db_{0}.png'.format(data_nm))
    plt.close()
