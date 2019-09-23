import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 


def draw_Point_Cloud(Points, Lables, axis = True, **kags):
    #%matplotlib inline

    x_axis = Points[:,0]
    y_axis = Points[:,1]
    z_axis = Points[:,2]
    fig = plt.figure() 
    ax = Axes3D(fig) 

    ax.scatter(x_axis, y_axis, z_axis, c = Lables)
    # 设置坐标轴显示以及旋转角度
    ax.set_xlabel('x') 
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.view_init(elev=10,azim=235)
    if not axis:
        #关闭显示坐标轴
        plt.axis('off')
    
    plt.show()

def NormalizedStd(data):
    data_mean = np.mean(data, axis = 0)
    data_std = np.std(data, axis = 0)
    return (data - data_mean) / data_std
    
#读取数据
data_feature = pd.read_csv('features.csv')

#数据预处理
from sklearn import preprocessing
np_data = NormalizedStd(data_feature.values)
min_max_scaler = preprocessing.MinMaxScaler()
#np_data = preprocessing.StandardScaler().fit_transform(data_feature.values)
np_data = min_max_scaler.fit_transform(data_feature.values)

#PCA降维
#from sklearn.decomposition import PCA
#pca = PCA(n_components=3)
# np_data_3d = pca.fit(np_data)
# #返回所保留的n个成分各自的方差百分比
# print(pca.explained_variance_ratio_)
# print(pca.explained_variance_)
# 核 PCA 

from sklearn.decomposition import KernelPCA

pca = KernelPCA(n_components=6,kernel='rbf',gamma=15)
np_data_3d = pca.fit(np_data)



data_new_3d = pca.transform(np_data)

#显示处理后数据大小
print(data_new_3d.shape)

#各种聚类算法、评价 metrics
#k-means
from sklearn.cluster import KMeans
import sklearn.metrics as metrics
from sklearn.cluster import DBSCAN





## k-means++ 
y_pred = KMeans(n_clusters=3, random_state=9).fit_predict(data_new_3d)

# y_pred = DBSCAN(eps=0.4,  # 邻域半径
# min_samples=5,    # 最小样本点数，MinPts
# metric='euclidean',
# metric_params=None,
# algorithm='auto', # 'auto','ball_tree','kd_tree','brute',4个可选的参数 寻找最近邻点的算法，例如直接密度可达的点
# leaf_size=10, # balltree,cdtree的参数
# p=None, # 
# n_jobs=1).fit_predict(data_new_3d)

print(metrics.calinski_harabaz_score(data_new_3d, y_pred))

#y_pred = KMeans(n_clusters=3, random_state=9).fit_predict(data_new_3d)

#print(metrics.calinski_harabaz_score(data_new_3d, y_pred))
# plt.scatter(X[:, 0], X[:, 1], c=y_pred)

# plt.show()
draw_Point_Cloud(data_new_3d, y_pred)