### 图像分类数据集


```python
%matplotlib inline
import d2lzh as d2l
from mxnet.gluon import data as gdata
import sys
import time
```

#### MNIST数据集是图像分类中广泛使用的数据集之一，但作为基准数据集过于简单。我们将使用类似但更复杂的Fashion-MNIST数据集。第一次调用会自动从网上获取数据。


```python
mnist_train = gdata.vision.FashionMNIST(root='../Fashion_MNIST_data',
    train=True,
    transform=None)
mnist_test = gdata.vision.FashionMNIST(root='../Fashion_MNIST_data',
    train=False,
    transform=None)
```

#### 通过框架中的内置函数将Fashion-MNIST数据集下载并读取到内存中


```python
len(mnist_train), len(mnist_test)
```




    (60000, 10000)




```python
feature, label = mnist_train[0]
```


```python
feature.shape, feature.dtype
```




    ((28, 28, 1), numpy.uint8)




```python
label, type(label), label.dtype
```




    (2, numpy.int32, dtype('int32'))




```python
# 本函数已经保存在d2lzh包中方便以后使用
def get_fashion_mnist_labels(labels):
    """将Fashion_MNIST数据集中的数值标签转换为文本标签。此数据集总共有10个类别，每个类别样本数相同"""
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat', 
                  'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]
```


```python
def show_fashion_mnist(images, labels):
    """在一行中话多张图像和对应的标签"""
    d2l.use_svg_display()
    # 这里的_表示我们忽略的变量
    _, figs = d2l.plt.subplots(1, len(images), figsize=(12,12))
    for f, img, lbl in zip(figs, images, labels):
        f.imshow(img.reshape((28, 28)).asnumpy())
        f.set_title(lbl)
        f.axes.get_xaxis().set_visible(False)
        f.axes.get_yaxis().set_visible(False)
```


```python
X, y = mnist_train[0:9]
show_fashion_mnist(X, get_fashion_mnist_labels(y))
```


    
![svg](output_11_0.svg)
    



```python
batch_size = 256
transformer = gdata.vision.transforms.ToTensor()
if sys.platform.startswith('win'):
    num_workers = 0  # 0表示不用额外的进程来加速读取数据
else:
    num_workers = 4

train_iter = gdata.DataLoader(mnist_train.transform_first(transformer),
                             batch_size, shuffle = True,
                             num_workers = num_workers)
test_iter = gdata.DataLoader(mnist_test.transform_first(transformer),
                            batch_size, shuffle = False,
                            num_workers = num_workers)
```


```python
start = time.time()
for X, y in train_iter:
    continue
'%.2f sec' % (time.time() - start)
```




    '7.11 sec'



### softmax回归从零开始实现


```python
def load_data_fashion_mnist(batch_size):
    mnist_train = gdata.vision.FashionMNIST(root='../Fashion_MNIST_data',
                       train=True, transform=None)
    mnist_test = gdata.vision.FashionMNIST(root='../Fashion_MNIST_data',
                        train=False, transform=None)
    transformer = gdata.vision.transforms.ToTensor()
    if sys.platform.startswith('win'):
        num_workers = 0
    else:
        num_workers = 4
    train_iter = gdata.DataLoader(mnist_train.transform_first(transformer),
                                 batch_size, shuffle = True,
                                 num_workers = num_workers)
    test_iter = gdata.DataLoader(mnist_test.transform_first(transformer),
                                batch_size, shuffle = False,
                                num_workers = num_workers)
    return train_iter, test_iter
```


```python
%matplotlib inline
import d2lzh as d2l
from mxnet import autograd, nd
```


```python
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
```

    Downloading C:\Users\CherishIntention\.mxnet\datasets\fashion-mnist\train-images-idx3-ubyte.gz from https://apache-mxnet.s3-accelerate.dualstack.amazonaws.com/gluon/dataset/fashion-mnist/train-images-idx3-ubyte.gz...
    Downloading C:\Users\CherishIntention\.mxnet\datasets\fashion-mnist\train-labels-idx1-ubyte.gz from https://apache-mxnet.s3-accelerate.dualstack.amazonaws.com/gluon/dataset/fashion-mnist/train-labels-idx1-ubyte.gz...
    Downloading C:\Users\CherishIntention\.mxnet\datasets\fashion-mnist\t10k-images-idx3-ubyte.gz from https://apache-mxnet.s3-accelerate.dualstack.amazonaws.com/gluon/dataset/fashion-mnist/t10k-images-idx3-ubyte.gz...
    Downloading C:\Users\CherishIntention\.mxnet\datasets\fashion-mnist\t10k-labels-idx1-ubyte.gz from https://apache-mxnet.s3-accelerate.dualstack.amazonaws.com/gluon/dataset/fashion-mnist/t10k-labels-idx1-ubyte.gz...
    

#### 初始化模型参数
#### 使用向量表示每个样本。已知样本输入是高和宽均为28像素的图像，模型输入向量的长度为28*28 = 784. 由于输出有10个类别，单层神经网络输出层的输出个数为10，因此softmax回归的权重和偏差参数分别为784X10和1X10的矩阵


```python
num_inputs = 784
num_outputs = 10

W = nd.random.normal(scale = 0.01, shape=(num_inputs, num_outputs))
b = nd.zeros(num_outputs)

W.attach_grad()
b.attach_grad()
```

#### 实现softmax运算


```python
def softmax(X):
    X_exp = X.exp()
    partition = X_exp.sum(axis=1, keepdims=True)
    return X_exp/partition  # 广播
```

#### 定义softmax回归模型


```python
def net(X):
    return softmax(nd.dot(X.reshape((-1, num_inputs)), W)+b)
```

#### 定义损失函数


```python
def cross_entropy(y_hat, y):
    return -nd.pick(y_hat, y).log()  # 输出类别互斥，损失函数可简化为-log(y_hat), y_hat正确类别的置信度
```


```python
# pick函数的使用
y_hat = nd.array([[0.1, 0.3, 0.6],[0.3, 0.2, 0.5]])
y = nd.array([0, 2], dtype='int32')
nd.pick(y_hat, y)
```




    
    [0.1 0.5]
    <NDArray 2 @cpu(0)>



#### 计算分类准确率


```python
def accuracy(y_hat, y):
    """正确预测的数量与总预测数之比"""
    return (y_hat.argmax(axis=1) == y.astype('float32')).mean().asscalar()
```


```python
def evaluate_accuracy(data_iter, net):
    acc_sum, n = 0.0, 0
    for X, y in data_iter:
        y = y.astype('float32')
        acc_sum += (net(X).argmax(axis=1) == y).sum().asscalar()
        n+=y.size
        return acc_sum / n
```


```python
evaluate_accuracy(test_iter, net)
```




    0.06640625




```python
next(iter(train_iter))[0].shape
```




    (256, 1, 28, 28)



#### 训练模型


```python
def train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size,
             params=None, lr=None, trainer=None):
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
        for X, y in train_iter:
            with autograd.record():
                y_hat = net(X)
                l = loss(y_hat, y).sum()
            l.backward()
            if trainer is None:
                d2l.sgd(params, lr, batch_size)
            else:
                trainer.step(batch_size)  # "softmax回归的简洁实现"一节将用到
            y = y.astype('float32')
            train_l_sum += l.asscalar()
            train_acc_sum += (y_hat.argmax(axis=1) == y).sum().asscalar()
            n += y.size
        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f'
             %(epoch+1, train_l_sum/n, train_acc_sum/n, test_acc))
```


```python
loss = cross_entropy
```


```python
train_ch3(net, train_iter, test_iter, loss, 5, batch_size, 
         [W, b], 0.1)
```

    epoch 1, loss 0.7888, train acc 0.749, test acc 0.820
    epoch 2, loss 0.5733, train acc 0.812, test acc 0.832
    epoch 3, loss 0.5281, train acc 0.824, test acc 0.848
    epoch 4, loss 0.5057, train acc 0.830, test acc 0.828
    epoch 5, loss 0.4898, train acc 0.834, test acc 0.844
    

#### 对图像进行分类


```python
for X, y in test_iter:
    break

true_labels = d2l.get_fashion_mnist_labels(y.asnumpy())
pred_labels = d2l.get_fashion_mnist_labels(net(X).argmax(axis=1).asnumpy())
titles = [true+'\n'+pred for true, pred in zip(true_labels, pred_labels)]
d2l.show_fashion_mnist(X[0:9], titles[0:9])
```


    
![svg](output_37_0.svg)
    

