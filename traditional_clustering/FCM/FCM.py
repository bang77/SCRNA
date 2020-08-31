import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import evaluate
def d(x, y, t):
    if t == 'vector':
        distance = np.linalg.norm(x - y)
    elif t == 'matrix':
        row, col = y.shape
        distance = np.zeros((row, 1))
        for i in range(row):
           distance[i] = np.linalg.norm(x - y[i,:])
    return distance
def FCM1(data,class_num,data_nm,label):
    # Hyper Parameters
    # C = int(class_num)
    C = class_num
    m = 1.1
    iteration = 10
    X = data
    n, dimension = X.shape
    # print(n)
    # print(dimension)
    U = np.array(np.random.rand(n, C), dtype='double')
    # print(U)
    U_crisp = np.zeros((n, 1))
    mu = np.zeros((C, dimension))
    # print(mu)
    X = np.array(X)
    fig, ax = plt.subplots()

    for k in range(iteration):

        for i in range(n):
            U[i,:] = U[i,:] / sum(U[i,:])

        for j in range(C):
            temp = (U[:,j] ** m)
            mu[j, :] = sum(np.multiply(temp,X.transpose()).transpose()) / sum(temp)

        for i in range(n):
            for j in range(C):
                U[i,j] = 1 / sum((d(X[i,:], mu[j,:], 'vector')) / d(X[i,:], mu[:,:], 'matrix')) ** (1 / (m-1))
    UV=[]
    for i in range(n):
       U_crisp[i] = np.argmax(U[i,:])
       UV.extend(U_crisp[i])
    # print(UV)
    print("标准化互信息      精度      纯度     轮廓系数    兰德系数")
    nmi, acc, purity,Sc,ARI = evaluate.eva(UV, label,data)
    print(nmi, acc, purity,Sc,ARI)
    colors = ['#B03060', '#AEEEEE', '#68228B', 'y', 'c', 'm','#2E2E2E','#00008B','#2E8B57','#8B0000','#8B5A00','#EEEE00','#CDCDB4','#ABABAB','#8B8B00']

    for i in range(C):
        points = np.array([X[j, :] for j in range(n) if U_crisp[j] == i])
        # print(points)
        ax.scatter(points[:, 0], points[:, 1], s=7, c=colors[i])
    plt.title("FCM+"+data_nm)
    # plt.legend(loc='best')
    plt.savefig('.\picture\FCM\F_{0}.png'.format(data_nm))
    # ax.scatter(mu[:, 0], mu[:, 1], marker='*', s=200, c='#ffff00')
    # plt.show()