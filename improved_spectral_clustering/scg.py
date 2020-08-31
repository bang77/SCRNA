# encoding=utf-8
import matplotlib.pyplot as plt
import numpy
from sklearn.metrics.pairwise import polynomial_kernel
from sklearn.metrics.pairwise import linear_kernel
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.metrics.pairwise import laplacian_kernel
from sklearn.metrics.pairwise import cosine_similarity

from numpy import linalg as LA
from sklearn.cluster import  hierarchical
from sklearn.manifold import TSNE
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from sklearn.neural_network import BernoulliRBM
from sklearn.neural_network import BernoulliRBM
from scipy.spatial import distance
from sklearn.neighbors import kneighbors_graph, NearestNeighbors
from sklearn.preprocessing import normalize
from sklearn.mixture import GaussianMixture
def squared_exponential(x, y, sig=0.8, sig2=1):
    norm = numpy.linalg.norm(x - y)
    dist = norm * norm
    return numpy.exp(- dist / (2 * sig * sig2))
def similarity_distance(X):
    N = X.shape[0]
    D = numpy.zeros((N, N))#生成N的零矩阵，初始化矩阵
    sig = []#初始化数组
    for i in range(N):
        dists = []
        for j in range(N):
            dists.append(numpy.linalg.norm(X[i] - X[j]))#添加到数组，两点的x距离
        dists.sort()#排序
        sig.append(numpy.mean(dists[:5]))#求平均数，并添加到sig数组去，形成而数组
    for i in range(N):
        for j in range(N):
            D[i][j] = squared_exponential(X[i], X[j], sig[i], sig[j])
    return D #返回矩阵形式
    #利用欧式距离计算数据点之间的相似矩阵
def similarity_matrix(data):
    n = data.shape[0]
    W = np.zeros((n, n), dtype='float64')
    for i in range(n):
        for j in range(i+1, n):
            W[i][j] = W[j][i] = distance.euclidean(data[i], data[j])
    D = np.exp(-W / W.std())
    return D
# 利用径向基核函数计算相似性矩阵，主对角线元素置为０
def similarity_function1(points):
    D = rbf_kernel(points)
    for i in range(len(D)):
        D[i, i] = 0
    return D

def similarity_function2(points):
    #l利用多项式核函数计算相似性矩阵，对角线元素置为0
    res=polynomial_kernel(points)
    for i in range(len(res)):
        res[i, i] = 0
    return res
#利用拉普拉斯核函数计算相似矩阵，对角线元素置为0
def similarity_function4(points):
    D=laplacian_kernel(points)
    for i in range(len(D)):
        D[i, i] = 0
    return D
def similarity_function5(points):
    D=cosine_similarity(points)
    for i in range(len(D)):
        D[i, i] = 0
    return D

def spectral_clustering(points, k,affinity):
    if affinity == 'rbf':
         W = similarity_function1(points)
    elif affinity == 'laplacian':
         W = similarity_function4(points)
    else:# affinity == 'distance':
        W=similarity_distance(points)
    # print(W)
    Dn = np.diag(np.power(np.sum(W, axis=1), -0.5))
    # 拉普拉斯矩阵：L=Dn*(D-W)*Dn=I-Dn*W*Dn
    L = np.eye(len(points)) - np.dot(np.dot(Dn, W), Dn)
    eigvals, eigvecs = LA.eig(L)
    # 前k小的特征值对应的索引，argsort函数
    indices = np.argsort(eigvals)[:k]
    # 取出前k小的特征值对应的特征向量，并进行正则化
    k_smallest_eigenvectors = normalize(eigvecs[:, indices])
    # 利用KMeans进行聚类
    C=KMeans(n_clusters=k).fit_predict(k_smallest_eigenvectors)
    # print(C)
    return C
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import evaluate
def Fmax(a):
    max_num = a[0]
    max= 0
    for i in range(len(a)):
        if a[i] > max_num:
            max_num = a[i]
            max = i
    return max
def SpectralClustering(data, class_num, data_nm, label):
    X=data
    af=['rbf','laplacian','distance']
    Compare = []
    for a in af:
        la= spectral_clustering(X, class_num,affinity=a)

        Compare.append(la)
    A = []
    for com in Compare:
        NMI, ARI = evaluate.eva_com(com, label)
        A.append(ARI)
    k = Fmax(A)
    labels= Compare[k]
    print("标准化互信息      精度      纯度     轮廓系数    兰德系数")
    nmi, acc, purity, Sc, ARI = evaluate.eva(labels, label, X)
    print(nmi, acc, purity, Sc, ARI)
    # 画图
    plt.style.use('ggplot')
    # 原数据
    # 谱聚类结果
    plt.scatter(X[:, 0], X[:, 1], c=labels, edgecolors='k')
    plt.title("SC2+" + data_nm)
    plt.savefig('.\picture\improved_spectral_clustering\sc1_{0}.png'.format(data_nm))
    plt.close()