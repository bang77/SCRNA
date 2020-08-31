import numpy as np
from sklearn.neighbors import NearestNeighbors
from itertools import combinations
import scipy.sparse.linalg
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from sklearn.cluster import DBSCAN
# from sklearn.cluster import
def filter(X):
    [rows, cols] = X.shape
    H=[]
    for i in range(rows):
        a=0
        for j in range(cols):
            if  int(X[i][j])>0:
                a=a+1
        if a>4:
            H.append(X[i])
    data=np.array(H)
    # [rows1, cols1]= data.shape
    # L=data[:]
    # for i in range(cols1):
    #     a=0
    #     for j in range(rows1):
    #         # print(data[j][i])
    #         if  int(data[j][i])>0:
    #             a=a+1
    #     if a<40:
    #          L= np.delete(L, i, axis=1)
    # hh=np.array(L)
    return data
def la(data):
    ms = SpectralClustering(affinity="laplacian").fit(data)
    label =ms.labels_
    return label
# nearest_neighbors
def snn_sim_matrix(X, k):
    """
    利用sklearn包中的KDTree,计算节点的共享最近邻相似度(SNN)矩阵
   """
    try:
        X = np.array(X)
    except:
        raise ValueError("输入的数据集必须为矩阵")
    samples_size, features_size = X.shape  # 数据集样本的个数和特征的维数
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='kd_tree').fit(X)
    knn_matrix = nbrs.kneighbors(X, return_distance=False)  # 记录每个样本的k个最近邻对应的索引
    sim_matrix = 0.5 + np.zeros((samples_size, samples_size))  # snn相似度矩阵
    for i in range(samples_size):
        t = np.where(knn_matrix == i)[0]
        c = list(combinations(t, 2))
        for j in c:
            if j[0] not in knn_matrix[j[1]]:
                continue
            sim_matrix[j[0]][j[1]] += 1
    sim_matrix = 1 / sim_matrix  # 将相似度矩阵转化为距离矩阵
    sim_matrix = np.triu(sim_matrix)
    sim_matrix += sim_matrix.T - np.diag(sim_matrix.diagonal())
    return sim_matrix
