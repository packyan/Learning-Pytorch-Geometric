# 深度学习——Pytorch笔记

## 主要内容
主要内容基本上都是李沐老师的[《动手学深度学习》](https://zh.d2l.ai/)中的内容，然后这个[github仓库](https://github.com/ShusenTang/Dive-into-DL-PyTorch)将书中的mxnet代码改为了Pytorch代码。
内容基本还是一致的，原仓库里面的内容比较详尽，这里只是把一些重点拿出来了，再加上一些我自己认为比较重要的内容。

## Pytorch基础部分

[Tensor的基本操作](2_2_Tensor.ipynb)

[自动求导机制 AutoGrad](2_3_AutoGrad.ipynb)


## 机器学习基础部分

[线性回归](3_1_Linear-regression.ipynb)

[softmax 回归](3_6_7_Softmax.ipynb)

[多层感知机 MLP](3_9_10_MLP.ipynb)

[模型复杂度、欠拟合、过拟合](3_11_overfitting.ipynb)

权重衰减 丢弃法 正向传播、反向传播和计算图 数值稳定性 和 模型初始化[可以查看原始 github仓库](https://github.com/ShusenTang/Dive-into-DL-PyTorch)

[Kaggle 房价预测](3_16_Kaggle-house-prices.ipynb)

## 卷积神经网络基础

[模型构造(如何搭建神经网络模型的结构、模型参数的初始化，访问、共享)](4_1_4_Creat_Modules.ipynb)

[储存、读取数据与模型](4_5_Save_Load.ipynb)

[GPU计算](4_6_Pytorch_GPU.ipynb)

[卷积层、池化层](Convolutional_Neural_Network.ipynb)

## 经典的卷积神经网络模型

[LeNet 第一个手写字符识别卷积神经网络](LeNet.ipynb)

[AlexNet 深度学习应用于CV领域的开山之作](AlexNet.ipynb)

[使用重复元素的网络（VGG）](VGG.ipynb)

网络中的网络（NiN）

[含并行连结的网络（GoogLeNet）](GoogleLeNet-Inception.ipynb)

[批量归一化](batch_normalization.ipynb)

[残差网络（ResNet）](ResNet.ipynb)

[稠密连接网络（DenseNet）](DenseNet.ipynb)

## 数据载入

[Dataloader与h5py文件](DataLoader.ipynb)

## 优化算法

[常见的损失函数](Loss_Function.ipynb)

[优化器](Optimizer.ipynb)


## 模型的评估方法

[ROC曲线 PR曲线 AUC AP mAP等](Interviews/ROC&PR&AUC&AP&mAP.ipynb)


## Pytorch Gotchas

[常见的坑](Pytorch_Gotchas.ipynb)


