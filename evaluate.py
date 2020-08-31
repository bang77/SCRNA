
import math
import numpy as np
from sklearn import metrics

def eva(A, B,data):
    total = len(A)#行数
    A_ids = set(A)#创建集合，原数据和模型的集合
    B_ids = set(B)
    eps = 1.4e-45
    purity = 0
    for idA in A_ids:
        max_purity = 0.0
        for idB in B_ids:
            idAOccur = np.where(A == idA)                     # 符合条件的行数返回下标
            idBOccur = np.where(B == idB)                     # 符合条件的行数返回下标
            idABOccur = np.intersect1d(idAOccur, idBOccur)#交集，并返回有序结果
            px = 1.0*len(idAOccur[0])/total
            py = 1.0*len(idBOccur[0])/total
            pxy = 1.0*len(idABOccur)/total
            if len(idABOccur) > max_purity:                  # 纯度计算
                max_purity = len(idABOccur)
                purity = purity + 1.0*len(idABOccur)/total
    acc=metrics.accuracy_score(A,B)
    # 标准化互信息
    NMI = metrics.normalized_mutual_info_score(A, B)
    SC=metrics.silhouette_score(data, B, metric='euclidean')
    ARI=metrics.adjusted_rand_score(A, B)
    return NMI, acc, purity,SC,ARI
def eva_com(A, B):
    NMI = metrics.normalized_mutual_info_score(A, B)
    ARI=metrics.adjusted_rand_score(A, B)
    return NMI,ARI