from scipy.io import loadmat
import numpy as np
from sklearn.manifold import TSNE
import pandas as pd
from sklearn.utils import shuffle
#载入数据,返回每个数据集含有的参数
def load_mat(name):
    mat_data = loadmat('data/' +name)
    class_num= mat_data['class_num']
    class_num=int(class_num)
    data = mat_data['data']
    label = mat_data['label'].reshape(-1)
    return data,class_num,label,name
