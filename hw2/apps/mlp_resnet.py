import sys

sys.path.append("../python")
import needle as ndl
import needle.nn as nn
import numpy as np
import time
import os

np.random.seed(0)
# MY_DEVICE = ndl.backend_selection.cuda()


def ResidualBlock(dim, hidden_dim, norm=nn.BatchNorm1d, drop_prob=0.1):
    ### BEGIN YOUR SOLUTION
    # 残差块结构：
    # Linear(dim, hidden_dim) -> norm -> ReLU -> Dropout -> Linear(hidden_dim, dim) -> norm -> ReLU
    # 包装在 Residual 中实现跳跃连接
    return nn.Sequential(
        nn.Residual(
            nn.Sequential(
                nn.Linear(dim, hidden_dim),
                norm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(drop_prob),
                nn.Linear(hidden_dim, dim),
                norm(dim)
            )
        ),
        nn.ReLU()  # 残差连接后的 ReLU
    )
    ### END YOUR SOLUTION


def MLPResNet(
    dim,
    hidden_dim=100,
    num_blocks=3,
    num_classes=10,
    norm=nn.BatchNorm1d,
    drop_prob=0.1,
):
    ### BEGIN YOUR SOLUTION
    # MLPResNet 结构：
    # Flatten -> Linear(dim, hidden_dim) -> ReLU -> 
    # num_blocks * ResidualBlock(hidden_dim, hidden_dim//2) ->
    # Linear(hidden_dim, num_classes)
    
    layers = [
        nn.Flatten(),
        nn.Linear(dim, hidden_dim),
        nn.ReLU()
    ]
    
    # 添加 num_blocks 个残差块
    for _ in range(num_blocks):
        layers.append(ResidualBlock(hidden_dim, hidden_dim // 2, norm, drop_prob))
    
    # 最后的分类层
    layers.append(nn.Linear(hidden_dim, num_classes))
    
    return nn.Sequential(*layers)
    ### END YOUR SOLUTION


def epoch(dataloader, model, opt=None):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    # 设置模型模式
    if opt is not None:
        model.train()
    else:
        model.eval()
    
    loss_func = nn.SoftmaxLoss()
    total_loss = 0.0
    total_error = 0.0
    total_samples = 0
    
    for batch in dataloader:
        X, y = batch
        batch_size = X.shape[0]
        
        # 前向传播
        logits = model(X)
        loss = loss_func(logits, y)
        
        # 计算错误率
        predictions = np.argmax(logits.numpy(), axis=1)
        errors = np.sum(predictions != y.numpy())
        
        if opt is not None:
            # 训练模式：反向传播和参数更新
            opt.reset_grad()
            loss.backward()
            opt.step()
        
        total_loss += loss.numpy() * batch_size
        total_error += errors
        total_samples += batch_size
    
    # 计算平均值
    avg_error = total_error / total_samples
    avg_loss = total_loss / total_samples
    
    return avg_error, avg_loss
    ### END YOUR SOLUTION


def train_mnist(
    batch_size=100,
    epochs=10,
    optimizer=ndl.optim.Adam,
    lr=0.001,
    weight_decay=0.001,
    hidden_dim=100,
    data_dir="data",
):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    # 加载数据集
    train_dataset = ndl.data.MNISTDataset(
        os.path.join(data_dir, "train-images-idx3-ubyte.gz"),
        os.path.join(data_dir, "train-labels-idx1-ubyte.gz")
    )
    test_dataset = ndl.data.MNISTDataset(
        os.path.join(data_dir, "t10k-images-idx3-ubyte.gz"),
        os.path.join(data_dir, "t10k-labels-idx1-ubyte.gz")
    )
    
    train_dataloader = ndl.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = ndl.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # 创建模型和优化器
    model = MLPResNet(784, hidden_dim)
    opt = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # 训练
    for _ in range(epochs):
        train_error, train_loss = epoch(train_dataloader, model, opt)
    
    # 评估
    test_error, test_loss = epoch(test_dataloader, model, None)
    
    return train_error, train_loss, test_error, test_loss
    ### END YOUR SOLUTION


if __name__ == "__main__":
    train_mnist(data_dir="../data")
