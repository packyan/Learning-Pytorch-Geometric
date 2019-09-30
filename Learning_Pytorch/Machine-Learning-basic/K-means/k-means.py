import numpy as np
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
def NormalizedStd(data):
    data_mean = np.mean(data, axis = 0)
    data_std = np.std(data, axis = 0)
    return (data - data_mean) / data_std
    
raw_data_file = open('iris.txt','r')
raw_data = raw_data_file.readlines()
N = 150 # data number
D = 4 # data dimesions
data_mat = np.zeros((N,D))
lable_mat = np.zeros((N,1))
arritubutes = []
data_len = D + 1 #data dim plus label
lable_dic = {'Iris-setosa' :0, 'Iris-versicolor':1, 'Iris-virginica':2}
k = 3 # k classes
for i, data in enumerate(raw_data):
    data_list = data.split(',')
    
    arritubutes = [float(x) for i, x in  enumerate(data_list) if(i < data_len - 1)]
    data_mat[i] = arritubutes
    lable_mat[i] = lable_dic[data_list[-1].strip('\n')]
    #print(data_mat[i])
    #data_mat.row_stack(data_list[:data_len-2])
    #lable_mat.row_stack(data_list[-1])
print("data shape: {} , lable shape : {}".format(data_mat.shape, lable_mat.shape))
data_mat = NormalizedStd(data_mat) 
# pick k centers
init_centers = np.zeros((k,D))
init_center_index = []
for i, index in enumerate(list(range(25,150,49))):
    print("init index:%d"%index)
    init_centers[i] = data_mat[index]
    init_center_index.append(index)
    
next_centers = init_centers.copy()

last_centers  = init_centers.copy()

print("init center:\n {}".format(init_centers))
# k-means part
class_mat = np.zeros(lable_mat.shape)

predict_label = np.zeros(lable_mat.shape,dtype = np.int32)
inter = 0
class_dic = {'0' :[init_center_index[0]], '1':[init_center_index[1]], '2':[init_center_index[2]]}
while(1):
    inter += 1
    last_centers = next_centers.copy()
    random_chose_index = list(range(data_mat.shape[0]))
    random.shuffle(random_chose_index)
    #strat interation
    for i in random_chose_index:
        last_centers_2 = next_centers.copy()
        closed_distance = np.inf
        closed_cls = np.inf
        for j in range(last_centers_2.shape[0]):
            now_distance = np.sum((data_mat[i] - last_centers_2[j])**2)
            if( now_distance < closed_distance):
                closed_distance = now_distance.copy();
                closed_cls = j
                # 第二次迭代开始，改变类别先从原有类中删除
        if(inter > 1):
            #print(class_dic)
            #print(predict_label)
            #print("try to remove {} from {}\n".format(i,predict_label[i][0]))
            class_dic[str(predict_label[i][0])].remove(i)
                       
        class_dic[str(closed_cls)].append(i)
        predict_label[i][0] = closed_cls
        #print(class_dic)
    # update center
    # center_temp = np.zeros((1,4))
        for i in range(k):
            center_temp = np.zeros((1,4))
            for index in (class_dic[str(i)]):
                center_temp += data_mat[index]
            next_centers[i] = center_temp/ (len(class_dic[str(i)]))
    #print("next")
    #print(next_centers)
    #print("last")
    #print(last_centers)
    #print(np.mean(last_centers - next_centers))
    ref_e = abs(np.mean(last_centers - next_centers))
    print("interation : {}\t rel error : {}".format(inter,ref_e))
    if(abs(ref_e) < 1e-5):
        print("0 class:{}\n 1 class:{}\n 2 class:{} \n".format(class_dic['0'], class_dic['1'], class_dic['2']))
        break


def draw_Point_Cloud(Points, Lables, axis = True, **kags):

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

draw_Point_Cloud(data_mat, predict_label.reshape(150), axis = False)
acc = []
for k in class_dic:
    correct = 0
    for ele in class_dic[k]:
        if( (int(k))*50 <= ele and (int(k)+1) * 50 > ele ):
            correct += 1
    acc.append(correct / len(class_dic[k]))
    print(acc)
#print(predict_label.reshape(150))
print("result is : \n{}".format(next_centers))
print(acc)
print("interation {} times".format(inter))

