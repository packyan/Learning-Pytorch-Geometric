import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
data_no = 1

N = 0 # data number
D = 0 # data dimesions
iris = 'iris.txt'
text_Data = 'text_data.txt'
if(data_no == 1):
    file_name = iris
    N = 150 # data number
    D = 4 # data dimesions
elif(data_no == 3):
    file_name = 'iris2.txt'
    N = 100
    D = 4
else:
    file_name = text_Data
    N = 21
    D = 2
raw_data_file = open(file_name,'r')
raw_data = raw_data_file.readlines()



data_mat = np.zeros((N,D))
lable_mat = np.zeros((N,1))
arritubutes = []
data_len = D + 1 #data dim plus label
lable_dic = {'Iris-setosa' :0, 'Iris-versicolor':1, 'Iris-virginica':2}
k = 3 # k classes
for i, data in enumerate(raw_data):
    #print(i)
    #print(data)
    data_list = data.split(',')
    arritubutes = [float(x) for i, x in  enumerate(data_list) if(i < D)]
    data_mat[i] = arritubutes
    #lable_mat[i] = lable_dic[data_list[-1].strip('\n')]
    #print(data_mat[i])
    #data_mat.row_stack(data_list[:data_len-2])
    #lable_mat.row_stack(data_list[-1])
print("data shape: {} , lable shape : {}".format(data_mat.shape, lable_mat.shape))
predict_label = np.zeros(lable_mat.shape,dtype = np.int32)
N_list = []
#Objective_E = np.array(150)
N_list.append(N)
N_list.append(0)
N_list.append(0)
init_data_index = list(range(0,N,1))
G_dic = {'0' : init_data_index, '1' : [], '2' :[]}

#current_G1_mat = data_mat.copy()
#current_G2_mat = np.zeros(data_mat.shape)

# x_mean_1 = np.mean(current_G1_mat, axis = 0) # x^(k+1)
# x_mean_2 = np.mean(current_G2_mat, axis = 0)
# print(x_mean_1)
# print(x_mean_2)

max_E_index = np.inf
max_E_last = 0
max_E = 0
iter = 0
# new idea use a dic{[]}  to store 3 classes' index
#[0~149] [] [] , then change this 
#for k_i in range(k - 1):
#print(G_dic[str(0)])
#print(len(G_dic[str(0)]))
for k in range(2):
    iter += 1
    max_E_index = np.inf
    max_E_last = 0
    max_E = 0
    print("data mat shape {}".format(data_mat.shape))
    #print(x_mean_1)
    #print(x_mean_2)

    while(1):
        current_G1_mat = data_mat[G_dic[str(k)],: ].copy().reshape((len(G_dic[str(k)]), D))
        current_G2_mat = data_mat[G_dic[str(k+1)],:].copy()
        if(not len(G_dic[str(k+1)]) == 0):
            current_G2_mat = current_G2_mat.reshape((len(G_dic[str(k+1)]), D))
        else:
            current_G2_mat = np.zeros((1,D))
        if(iter > 2):
            print(current_G1_mat.shape)
            print(current_G2_mat)
        #print("G1 mat:{}".format(current_G1_mat))
        #print("G2 mat :{}".#format(current_G2_mat))
        #print(current_G1_mat)
        #print( np.mean(current_G1_mat, axis = 0))
        x_mean_1 = np.mean(current_G1_mat, axis = 0) # x^(k+1)
        x_mean_2 = np.mean(current_G2_mat, axis = 0)
        #print("x k mean : ")
        #print(x_mean_1)
        #print(x_mean_2)
        #print("***********")
        #max_E = len(G_dic[str(k)])*len(G_dic[str(k+1)]) / (len(G_dic[str(k)]) + len(G_dic[str(k+1)]))*((x_mean_1 - x_mean_2).T) * (x_mean_1 - x_mean_2)
        #max_E = np.sqrt(np.sum(max_E ** 2))
        max_E = len(G_dic[str(k)])*len(G_dic[str(k+1)]) / (len(G_dic[str(k)]) + len(G_dic[str(k+1)]))*np.sum((x_mean_1 - x_mean_2)**2)
        #print("current E:  {}".format(max_E))
        max_E_index = np.inf
        if(iter >2):
            print("max E init :{}".format(max_E))
        for i, x_index in enumerate(G_dic[str(k)]):
            #iter += 1
            x_data = data_mat[x_index]
            #print("x:{} index: {}".format(x_data,x_index))
            G_dic_temp = deepcopy(G_dic)
            G_dic_temp[str(k)].remove(x_index)
            G_dic_temp[str(k+1)].append(x_index)
            # temp_G1_mat = data_mat[G_dic_temp[str(k)], :].copy()
            # temp_G2_mat = data_mat[G_dic_temp[str(k+1)] , :].copy()
            # temp_G1_mat = temp_G1_mat.reshape((len(G_dic_temp[str(k)]), D))
            # temp_G2_mat = temp_G2_mat.reshape((len(G_dic_temp[str(k+1)]), D))
            #print(temp_G1_mat)
            #print(temp_G2_mat)
            #print(temp_G1_mat.shape)
            #print(temp_G2_mat)
            x_mean_1_temp = x_mean_1 + (x_mean_1 - x_data)/ (len(G_dic[str(k)]) -1)
            x_mean_2_temp = x_mean_2 - (x_mean_2 - x_data)/ ( len(G_dic[str(k+1)]) + 1)
            #print(x_mean_1_temp)
            # if(len(G_dic_temp[str(k+1)]) > 1):
                # x_mean_2_temp = x_mean_2 - (x_mean_2 - x_data)/ ( len(G_dic_temp[str(k+1)]) - 1)
            # else:
                # x_mean_2_temp = x_data
            if(iter > 2):
                
            #print(x_mean_2_temp)
                print("x mean: ")
                print(x_mean_1_temp)
                print(x_mean_2_temp)
            #current_E = (len(G_dic_temp[str(k)]))*len(G_dic_temp[str(+1)] )/(len(G_dic_temp[str(k)]) + len(G_dic_temp[str(k+1)]) )*(x_mean_1_temp - x_mean_2_temp).T * (x_mean_1_temp - x_mean_2_temp)
            #print(current_E)
            #current_E = np.sqrt(np.sum(current_E ** 2))
            current_E = (len(G_dic_temp[str(k)]))*len(G_dic_temp[str(k+1)] )/(len(G_dic_temp[str(k)]) + len(G_dic_temp[str(k+1)]) ) *np.sum( (x_mean_1_temp - x_mean_2_temp)**2)
            if(iter > 2):
                print( (len(G_dic_temp[str(k)]), len(G_dic_temp[str(k+1)])))
                print((len(G_dic_temp[str(k)]))*len(G_dic_temp[str(k+1)] )/(len(G_dic_temp[str(k)]) + len(G_dic_temp[str(k+1)]) ))
                print(np.sum( (x_mean_1_temp - x_mean_2_temp)**2))
                print(current_E)
            #print(current_E)
            if current_E > max_E:
                max_E = current_E
                max_E_index = x_index
                
        
        if(max_E < max_E_last or max_E_index == np.inf):
            print(" ********\nfinished : max_E :{}, last_max_E:{}\n************".format(max_E, max_E_last))
            break;
        else:
            #print("\n\ninter: {}, add {}\n\n".format(iter, max_E_index))
            max_E_last = max_E
            #print(G_dic[str(k)])
            G_dic[str(k)].remove(max_E_index)
            G_dic[str(k+1)].append(max_E_index)
            #print(G_dic[str(k)])
            #print(G_dic[str(k+1)])
            #sample_mean = np.mean(current_data_mat, axis = 0)
            #
print(G_dic)


for k in G_dic:
    for ele in G_dic[k]:    
        predict_label[ele] = k
    
    
x_axis = data_mat[:,0]
y_axis = data_mat[:,1]
z_axis = data_mat[:,2]
fig = plt.figure() 
ax = Axes3D(fig) 

ax.scatter(x_axis, y_axis, z_axis, c=predict_label.reshape(150))
# 设置坐标轴显示以及旋转角度
ax.set_xlabel('1') 
ax.set_ylabel('2')
ax.set_zlabel('3')
ax.view_init(elev=10,azim=235)
plt.show()


acc = []
for k in G_dic:
    correct = 0
    for ele in G_dic[k]:
        if( (int(k))*50 <= ele and (int(k)+1) * 50 > ele ):
            correct += 1
    acc.append(correct / len(G_dic[k]))
print(acc)


