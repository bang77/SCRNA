import numpy as np
from filter import filter
from sklearn.manifold import TSNE
from sklearn import manifold
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
# from tsne import bh_sne
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.neural_network import BernoulliRBM
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from filter import snn_sim_matrix
from filter import la
def Gene_Zeisel():
    data_nm="Zeisel"
    X = pd.read_csv("data/GSE76381.txt", sep='\t', low_memory=False).T
    # clrs =['#B03060', '#AEEEEE', '#68228B', 'y', 'c', 'm', '#2E2E2E', '#00008B', '#2E8B57', '#FAEBD7',
    #                  '#8B5A00', '#EEEE00', '#0000FF', '#ABABAB', '#8B8B00']
    clusters = np.array(X[7], dtype=str)[2:]
    cell_types, label = np.unique(clusters, return_inverse=True)
    data= np.array(X.iloc[2:, 11:],dtype=np.int)
    class_num = len(np.unique(label))
    # data = PCA(n_components=100).fit_transform(data)
    # data=snn_sim_matrix(data, k=class_num)
    # data= LinearDiscriminantAnalysis().fit_transform(data,label)
    # data=BernoulliRBM(n_components=50).fit_transform(data)
    data = TSNE(n_components=2,perplexity=42,angle=0.8).fit_transform(data)
    # print(data)
    # print(label)
    # print(class_num)
    # for k, clr in enumerate(clrs):
    #     cur = (label == k)
    #     plt.scatter(data[cur, 0], data[cur, 1], s=40, color=clr, edgecolors='k')
    # plt.show()

    return data,class_num,label,data_nm