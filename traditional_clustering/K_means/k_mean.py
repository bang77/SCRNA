import random
import numpy as np
from traditional_clustering.K_means import Kmeans
import evaluate


def kmeans(data,class_num,data_nm,label):
    k = class_num
    clu = random.sample(data.tolist(), k)  # 随机取质心
    clu = np.asarray(clu)
    err, clunew, k, clusterRes = Kmeans.classfy(data, clu, k)
    while np.any(abs(err) > 0):
        # print(clunew)
        err, clunew, k, clusterRes = Kmeans.classfy(data, clunew, k)

    clulist = Kmeans.cal_dis(data, clunew, k)
    clusterResult = Kmeans.divide(data, clulist)
    # print(clusterResult)
    # print(label.reshape(-1))
    print("标准化互信息      精度      纯度     轮廓系数    兰德系数")
    nmi, acc, purity,Sc,ARI= evaluate.eva(clusterResult, label,data)
    print(nmi, acc, purity,Sc,ARI)
    Kmeans.plotRes(data, clusterResult, k, data_nm)