import time
from improved_spectral_clustering import scg
from traditional_clustering.dbscan import dbscan
from traditional_clustering.K_means import k_mean
from traditional_clustering.hierarchical_clustering import hc
from traditional_clustering.FCM import FCM
from indata import load_mat
from Gene_treatment.gene1 import Gene_TH
from Gene_treatment. gene2 import Gene_Zeisel
t1=time.time()
dataset = ['banana', 'gaussian', 'ids2', 'lithuanian','Th+','Zeisel']
for i in range(len(dataset)):
    if dataset[i] in ['banana', 'gaussian', 'ids2', 'lithuanian']:
        data,class_num,label,data_nm=load_mat(dataset[i])
    elif dataset[i] in ['Th+']:
        data, class_num, label, data_nm = Gene_TH()
    else:
        data, class_num, label, data_nm = Gene_Zeisel()
    print("{}基于密度的聚类算法：".format(data_nm))
    dbscan.dbscan(data, data_nm,class_num, label)
    print("{}基于层次的聚类算法：".format(data_nm))
    hc.HC(data, class_num, data_nm, label)
    print("{}K均值的聚类算法：".format(data_nm))
    k_mean.kmeans(data, class_num, data_nm, label)
    print("{}基于模糊的聚类算法：".format(data_nm))
    FCM.FCM1(data, class_num, data_nm, label)
    print("{}谱聚类的聚类：".format(data_nm))
    scg.SpectralClustering(data, class_num, data_nm, label)

t2=time.time()
print("所用时间：{}".format(int(t2-t1)))