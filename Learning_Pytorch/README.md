# 深度学习——Pytorch笔记

## 主要内容
主要内容基本上都是李沐老师的[《动手学深度学习》](https://zh.d2l.ai/)中的内容，然后这个[github仓库](https://github.com/ShusenTang/Dive-into-DL-PyTorch)将书中的mxnet代码改为了Pytorch代码。
内容基本还是一致的，原仓库里面的内容比较详尽，这里只是把一些重点拿出来了，再加上一些我自己认为比较重要的内容。

## Pytorch基础部分

[Tensor的基本操作](Pytorch-basic/2_2_Tensor.ipynb)

[自动求导机制 AutoGrad](Pytorch-basic/2_3_AutoGrad.ipynb)



## 数据可视化

[matplotlib绘图](Visualization/matlib_plot.ipynb)


## 机器学习基础部分

[数据预处理sklearn](Machine-Learning-basic/Data-preprocessing.ipynb)

[数据降维：PCA 与 核PCA](Machine-Learning-basic/KPCA+PCA+SVD.ipynb)

[Kmeans，分解聚类](Machine-Learning-basic/Cluster.ipynb)

[特征工程](Machine-Learning-basic/Feature-Engineering.ipynb)


[线性回归](Machine-Learning-basic/3_1_Linear-regression.ipynb)

[softmax 回归](Machine-Learning-basic/3_6_7_Softmax.ipynb)

[多层感知机 MLP](Machine-Learning-basic/3_9_10_MLP.ipynb)

[模型复杂度、欠拟合、过拟合](Machine-Learning-basic/3_11_overfitting.ipynb)

权重衰减 丢弃法 正向传播、反向传播和计算图 数值稳定性 和 模型初始化[可以查看原始 github仓库](https://github.com/ShusenTang/Dive-into-DL-PyTorch)

[Kaggle 房价预测](Machine-Learning-basic/3_16_Kaggle-house-prices.ipynb)

## 卷积神经网络基础

[模型构造(如何搭建神经网络模型的结构、模型参数的初始化，访问、共享)](Convolutional-Neural-Networks/4_1_4_Creat_Modules.ipynb)

[储存、读取数据与模型](Convolutional-Neural-Networks/4_5_Save_Load.ipynb)

[GPU计算](Convolutional-Neural-Networks/4_6_Pytorch_GPU.ipynb)

[卷积层、池化层](Convolutional-Neural-Networks/Convolutional_Neural_Network.ipynb)

[转置卷积-deconv](Convolutional-Neural-Networks/deconv.ipynb)

[空洞卷积-dilated-conv](Convolutional-Neural-Networks/dilated-convolution.ipynb)

## 经典的卷积神经网络模型

[LeNet 第一个手写字符识别卷积神经网络](Classical-CNN-Architecture/LeNet.ipynb)

[AlexNet 深度学习应用于CV领域的开山之作](Classical-CNN-Architecture/AlexNet.ipynb)

[使用重复元素的网络（VGG）](Classical-CNN-Architecture/VGG.ipynb)

网络中的网络（NiN）

[含并行连结的网络（GoogLeNet）](Classical-CNN-Architecture/GoogleLeNet-Inception.ipynb)

[批量归一化](Classical-CNN-Architecture/batch_normalization.ipynb)

[残差网络（ResNet）](Classical-CNN-Architecture/ResNet.ipynb)

[稠密连接网络（DenseNet）](Classical-CNN-Architecture/DenseNet.ipynb)

## 循环神经网络

[Recurrent-Neural-Networks(GRU LSTM)](Recurrent-Neural-Networks/RNN.ipynb)


## 计算机视觉的应用

[数据增强](Computer-Vision/image-augmentation.ipynb)

[Bounding-box](Computer-Vision/Bounding-box.ipynb)

[微调](Computer-Vision/fine-tuning.ipynb)

[目标检测入门--SSD](Computer-Vision/a-PyTorch-Tutorial-to-Object-Detection.ipynb)

## 数据载入

[Dataloader与h5py文件](Pytorch-basic/DataLoader.ipynb)

[Pandas基础](Data-Preprocessing/10-minutes-to-pandas.ipynb)

## 优化算法

[常见的损失函数](Optimization/Loss_Function.ipynb)

[优化器](Optimization/Optimizer.ipynb)


## 模型的评估方法

[ROC曲线 PR曲线 AUC AP mAP等](Interviews/ROC&PR&AUC&AP&mAP.ipynb)


## Pytorch Gotchas

[常见的坑](Pytorch_Gotchas.ipynb)