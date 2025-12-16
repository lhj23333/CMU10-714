"""hw1/apps/simple_ml.py"""

import struct
import gzip
import numpy as np

import sys

sys.path.append("python/")
import needle as ndl


def parse_mnist(image_filename, label_filename):
    """ Read an images and labels file in MNIST format.  See this page:
    http://yann.lecun.com/exdb/mnist/ for a description of the file format.

    Args:
        image_filename (str): name of gzipped images file in MNIST format
        label_filename (str): name of gzipped labels file in MNIST format

    Returns:
        Tuple (X,y):
            X (numpy.ndarray[np.float32]): 2D numpy array containing the loaded
                data.  The dimensionality of the data should be
                (num_examples x input_dim) where 'input_dim' is the full
                dimension of the data, e.g., since MNIST images are 28x28, it
                will be 784.  Values should be of type np.float32, and the data
                should be normalized to have a minimum value of 0.0 and a
                maximum value of 1.0.

            y (numpy.ndarray[dypte=np.int8]): 1D numpy array containing the
                labels of the examples.  Values should be of type np.int8 and
                for MNIST will contain the values 0-9.
    """
    # 读取图像文件
    with gzip.open(image_filename, 'rb') as img_file:
        # 读取文件头： magic number，图像数量，行数，列数
        # '>4i'表示大端字节序的4个无符号整数
        magic_num, img_num, row, col = struct.unpack(">4i", img_file.read(16))
        assert(magic_num == 2051)

        # 读取所有图像数据
        tot_pixels = row * col

        # 将字节数据转换为numpy数组并归一化
        X = np.vstack([np.array(struct.unpack(f"{tot_pixels}B", img_file.read(tot_pixels)), dtype=np.float32) for _ in range(img_num)])
        X -= np.min(X)
        X /= np.max(X)

    # 读取标签文件
    with gzip.open(label_filename, 'rb') as label_file:
        # 读取文件头： magic number，标签数量
        magic_num, label_num = struct.unpack(">2i", label_file.read(8))
        assert(magic_num == 2049)

        # 将字节数据转换为numpy数组
        y = np.array(struct.unpack(f"{label_num}B", label_file.read()), dtype=np.uint8)

    return X, y


def softmax_loss(Z, y_one_hot):
    """Return softmax loss.  Note that for the purposes of this assignment,
    you don't need to worry about "nicely" scaling the numerical properties
    of the log-sum-exp computation, but can just compute this directly.

    Args:
        Z (ndl.Tensor[np.float32]): 2D Tensor of shape
            (batch_size, num_classes), containing the logit predictions for
            each class.
        y (ndl.Tensor[np.int8]): 2D Tensor of shape (batch_size, num_classes)
            containing a 1 at the index of the true label of each example and
            zeros elsewhere.

    Returns:
        Average softmax loss over the sample. (ndl.Tensor[np.float32])
    """
    # exp_Z = ndl.exp(Z)                              # 对所有 logits 取指数
    # sum_exp_per_sample = exp_Z.sum(axes=(1,))       # 对每个样本（行），累加所有类别 exp 和
    # log_sum_exp_per_sample = ndl.log(sum_exp_per_sample)    # 对每个样本的分母取对数
    # total_log_sum_exp = log_sum_exp_per_sample.sum()        # 累加所有样本 log(sum_exp) 和
    # true_class_logits_sum = (y_one_hot * Z).sum()           # 计算所有样本的真实类别 logits 和
    # total_loss = total_log_sum_exp - true_class_logits_sum  # 计算总损失

    # batch_size = Z.shape[0]
    # average_loss = total_loss / batch_size                 # 计算平均损失    
    
    # return average_loss

    return (ndl.log(ndl.exp(Z).sum((1,))).sum() - (y_one_hot * Z).sum()) / Z.shape[0]

def nn_epoch(X, y, W1, W2, lr=0.1, batch=100):
    """Run a single epoch of SGD for a two-layer neural network defined by the
    weights W1 and W2 (with no bias terms):
        logits = ReLU(X * W1) * W2
    The function should use the step size lr, and the specified batch size (and
    again, without randomizing the order of X).

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        W1 (ndl.Tensor[np.float32]): 2D array of first layer weights, of shape
            (input_dim, hidden_dim)
        W2 (ndl.Tensor[np.float32]): 2D array of second layer weights, of shape
            (hidden_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD mini-batch

    Returns:
        Tuple: (W1, W2)
            W1: ndl.Tensor[np.float32]
            W2: ndl.Tensor[np.float32]
    """
    iterations = (y.size + batch - 1) // batch      # 计算总的迭代次数，向上取整

    for i in range(iterations):
        x = ndl.Tensor(X[i*batch : (i + 1)*batch, :])   # 获取当前批次的输入数据
        z = ndl.relu(x.matmul(W1)).matmul(W2)           # 前向传播计算 logits

        # 创建并计算 one-hot 编码矩阵
        yy = y[i*batch : (i + 1)*batch]                 # 获取当前批次的标签
        y_one_hot = np.zeros((batch, y.max() + 1))      
        y_one_hot[np.arange(batch), yy] = 1
        y_one_hot = ndl.Tensor(y_one_hot)

        loss = softmax_loss(z, y_one_hot)               # 计算损失
        loss.backward()                                 # 反向传播计算梯度

        # 更新矩阵权重
        W1 = ndl.Tensor(W1.realize_cached_data() - lr*W1.grad.realize_cached_data())
        W2 = ndl.Tensor(W2.realize_cached_data() - lr*W2.grad.realize_cached_data())
    
    return W1, W2
### CODE BELOW IS FOR ILLUSTRATION, YOU DO NOT NEED TO EDIT


def loss_err(h, y):
    """Helper function to compute both loss and error"""
    y_one_hot = np.zeros((y.shape[0], h.shape[-1]))
    y_one_hot[np.arange(y.size), y] = 1
    y_ = ndl.Tensor(y_one_hot)
    return softmax_loss(h, y_).numpy(), np.mean(h.numpy().argmax(axis=1) != y)
