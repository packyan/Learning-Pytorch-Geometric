from IPython import display
from matplotlib import pyplot as plt
import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np

def set_figsize(figsize=(3.5, 2.5)):
    use_svg_display()
    # 设置图的尺寸
    plt.rcParams['figure.figsize'] = figsize

def use_svg_display():
    """Use svg format to display plot in jupyter"""
    display.set_matplotlib_formats('svg')
    
def linreg(X, w, b):
    return torch.mm(X, w) + b

def sgd(params, lr, batch_size):
    for param in params:
        param.data -= lr * param.grad / batch_size # 注意这里更改param时用的param.data

def squared_loss(y_hat, y):
     # 注意这里返回的是向量, 另外, pytorch里的MSELoss并没有除以 2
    return (y_hat-y.view(y_hat.size()))**2 /2

def data_creater(num_samples, features_num, true_w, true_b):
    torch.set_default_dtype(torch.float32)
    features = torch.from_numpy( np.random.normal(0,1,(num_samples,features_num)))
    #createfeatures
    #矩阵乘法形式
    labels = torch.mm(features,torch.tensor(true_w, dtype = torch.double).view(features_num,1))+true_b

    #labels_ = true_w[0]*features[:,0] + true_w[1]*features[:,1] + true_b
    labels = labels.view(num_samples)
    return (features).float() , (labels + torch.from_numpy(np.random.normal(0,0.01,labels.size()))).float()

def get_fashion_mnist_labels(labels):
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]
# 本函数已保存在d2lzh包中方便以后使用
def show_fashion_mnist(images, labels):
    use_svg_display()
    # 这里的_表示我们忽略（不使用）的变量
    _, figs = plt.subplots(1, len(images), figsize=(12, 12))
    for f, img, lbl in zip(figs, images, labels):
        f.imshow(img.view((28, 28)).numpy())
        f.set_title(lbl)
        f.axes.get_xaxis().set_visible(False)
        f.axes.get_yaxis().set_visible(False)
    plt.show()

def load_data_fashion_mnist(batch_size, num_workers = 4):
    mnist_train = torchvision.datasets.FashionMNIST(root='~/Datasets/FashionMNIST', train=True, download=True, transform=transforms.ToTensor())
    mnist_test = torchvision.datasets.FashionMNIST(root='~/Datasets/FashionMNIST', train=False, download=True, transform=transforms.ToTensor())
    train_iter = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_iter = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_iter, test_iter

# 本函数已保存在d2lzh_pytorch包中方便以后使用。该函数将被逐步改进：它的完整实现将在“图像增广”一节中描述
def evaluate_accuracy(data_iter, net):
    acc_sum, n = 0.0, 0
    for X, y in data_iter:
        acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
        n += y.shape[0]
    return acc_sum / n   

def train_classification_net(net, train_iter, test_iter, loss, num_epochs, batch_size, 
                             params=None, lr=None, optimizer=None):
    print('batch_size :{}'.format(batch_size))
    for epoch in range(num_epochs):

        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0

        for data, label in train_iter:

            out = net(data)

            l = loss(out,label).sum()

                    # 梯度清零
            if optimizer is not None:
                optimizer.zero_grad()
            elif params is not None and params[0].grad is not None:
                for param in params:
                    param.grad.data.zero_()

            l.backward()
            if optimizer is None:
                sgd(params, lr, batch_size)
            else:
                optimizer.step()  # “softmax回归的简洁实现”一节将用到

            #计算信息
            train_l_sum += l.item()
            train_acc_sum += (out.argmax(dim=1) == label).float().sum().item()
            n += label.shape[0]

        test_acc = evaluate_accuracy(test_iter, net)

        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f'
                  % (epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc))

# 本函数已保存在d2lzh_pytorch包中方便以后使用
class FlattenLayer(nn.Module):
    def __init__(self):
        super(FlattenLayer, self).__init__()
    def forward(self, x): # x shape: (batch, *, *, ...)
        return x.view(x.shape[0], -1)   

        
if __name__ == '__main__':
    features , labels = data_creater(100,3,[0.3,0.4,0.3],0.1)
    print(features.size(), features.dtype)
    print(labels.size())