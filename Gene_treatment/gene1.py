import pandas as pd
import numpy as np
from sklearn.decomposition import NMF
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import matplotlib.pyplot as plt
def Gene_TH():
    data_nm='Th+'
    # clrs =['#B03060', '#AEEEEE', '#68228B', 'y', 'c', 'm', '#2E2E2E', '#00008B', '#2E8B57', '#FAEBD7',
    #                  '#8B5A00', '#EEEE00', '#0000FF', '#ABABAB', '#8B8B00']
    X= pd.read_csv( "data/GSE76381_EmbryoMoleculeCounts.cef.txt", sep = '\t' ,low_memory=False).T
    # X=X.fillna(0)
    clusters = np.array(X[2], dtype=str)[5:3000]
    cell_types, label = np.unique(clusters, return_inverse=True)
    # label=np.array(X[2])[4:3000]
    class_num = len(np.unique(cell_types))
    data = np.array(X.iloc[5:3000, 4:])
    # data = LinearDiscriminantAnalysis().fit_transform(data, label)
    # data = NMF(n_components=50, init='random', random_state=0).fit_transform(data)
    # data = PCA(n_components=50).fit_transform(data)
    data=TSNE(n_components=2,perplexity=40,angle=0.8).fit_transform(data)
    # print(data)
    # print(label)
    # print(class_num)
    # for k, clr in enumerate(clrs):
    #     cur = (label == k)
    #     plt.scatter(data[cur, 0], data[cur, 1], s=40, color=clr, edgecolors='k')
    # plt.show()
    return data,class_num,label,data_nm
