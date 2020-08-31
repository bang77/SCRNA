import numpy as np
import math as m
import matplotlib.pyplot as plt
def cal_dis(data, clu, k):
    # 计算质点与数据点的距离
    dis = []
    for i in range(len(data)):
        dis.append([])
        for j in range(k):
            dis[i].append(m.sqrt((data[i, 0] - clu[j, 0])**2 + (data[i, 1]-clu[j, 1])**2))
    return np.asarray(dis)
def divide(data, dis):
    #对数据点分组
    clusterRes = [0] * len(data)
    for i in range(len(data)):
        seq = np.argsort(dis[i])
        clusterRes[i] = seq[0]
    return np.asarray(clusterRes)
def center(data, clusterRes, k):
    clunew = []
    for i in range(k):
        idx = np.where(clusterRes == i)
        sum = data[idx].sum(axis=0)
        avg_sum = sum/len(data[idx])
        clunew.append(avg_sum)
    clunew = np.asarray(clunew)
    return clunew[:, 0: 2]
def classfy(data, clu, k):
    clulist = cal_dis(data, clu, k)
    clusterRes = divide(data, clulist)
    clunew = center(data, clusterRes, k)
    err = clunew - clu
    return err, clunew, k, clusterRes
def plotRes(data, clusterRes, clusterNum,data_nm):
    nPoints = len(data)
    scatterColors = ['#B03060', '#AEEEEE', '#68228B', 'y', 'c', 'm','#2E2E2E','#00008B','#2E8B57','#8B0000','#8B5A00','#EEEE00','#CDCDB4','#ABABAB','#8B8B00']
    plt.figure()
    for i in range(clusterNum):
        color = scatterColors[i % len(scatterColors)]
        x1 = [];  y1 = []
        for j in range(nPoints):
            if clusterRes[j] == i:
                x1.append(data[j, 0])
                y1.append(data[j, 1])
        plt.scatter(x1, y1, c=color, s=40, edgecolors='k')
    plt.title("Kmeans+"+data_nm)
    # plt.legend(loc='best')
    # plt.savefig('D:\代码\zijidongshouban\picture\Kmeans\K_{0}.png'.format(data_nm))
    plt.savefig('.\picture\Kmeans\K_{0}.png'.format(data_nm))
    # plt.show()
    plt.close()
    # print("完成保存")