"""The module.
"""
from ast import Param
from typing import Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> list[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> list["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []


class Module:
    def __init__(self) -> None:
        self.training = True

    def parameters(self) -> list[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> list["Module"]:
        return _child_modules(self.__dict__)

    def eval(self) -> None:
        self.training = False
        for m in self._children():
            m.training = False

    def train(self) -> None:
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x


class Linear(Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, device: Any | None = None, dtype: str = "float32") -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # 初始化权重：使用 Kaiming_uniform
        self.weight = Parameter(
            init.kaiming_uniform(in_features, out_features, device=device, dtype=dtype)
        )

        # 初始化偏差
        if bias:
            # 注意：bias 的形状是 (1, out_features), 初始化时 fan_in=out_features
            self.bias = Parameter(
                init.kaiming_uniform(out_features, 1, device=device, dtype=dtype).reshape((1, out_features)) 
            )
        else:
            self.bias = None

    def forward(self, X: Tensor) -> Tensor:
        # 矩阵乘法：X @ W
        out = ops.matmul(X, self.weight)

        # 广播偏差：out + bias
        if self.bias is not None:
            out = out + ops.broadcast_to(self.bias, out.shape)

        return out

class Flatten(Module):
    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        # 计算展开后的特征维度：所有维度的乘积
        batch_size = X.shape[0]
        flatten_dim = 1
        for dim in X.shape[1:]:
            flatten_dim *= dim

        return ops.reshape(X, (batch_size, flatten_dim))
        ### END YOUR SOLUTION


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return ops.relu(x)


class Sequential(Module):
    def __init__(self, *modules: Module) -> None:
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        for module in self.modules:
            x = module(x)
        return x

class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        # logits: (batch_size, num_classes)
        # y: (batch_size,) - 标签（类别索引）

        batch_size = logits.shape[0]
        num_classes = logits.shape[1]

        # 计算 log(sum(exp(logits)))，沿着最后一个轴求和
        log_sum_exp = ops.logsumexp(logits, axes=(1,))  # (batch_size,)

        # 创建 one-hot 编码以提取正确类别的 logits
        y_one_hot = init.one_hot(num_classes, y)

        # 提取每个样本对应标签的 logit: z_{y_i}
        z_y = ops.summation(logits * y_one_hot, axes=(1,))  # (batch_size,)

        # 计算损失并求平均
        loss = ops.summation(log_sum_exp - z_y) / batch_size

        return loss
        ### END YOUR SOLUTION


class BatchNorm1d(Module):
    def __init__(self, dim: int, eps: float = 1e-5, momentum: float = 0.1, device: Any | None = None, dtype: str = "float32") -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        ### BEGIN YOUR SOLUTION
        # 可学习参数 gamma (weight) 和 beta (bias)
        self.weight = Parameter(
            init.ones(dim, device=device, dtype=dtype)
        ) 
        self.bias = Parameter(
            init.zeros(dim, device=device, dtype=dtype)
        )

        # 运行时统计量（不是参数，用 Tensor 存储）
        self.running_mean = init.zeros(dim, device=device, dtype=dtype)
        self.running_var = init.ones(dim, device=device, dtype=dtype)
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        # x: (batch_size, dim)
        batch_size = x.shape[0]
        
        if self.training:
            # 计算当前 batch 的均值和方差
            mean = ops.summation(x, axes=(0,)) / batch_size  # (dim,)
            # 广播 mean 到 x 的形状
            mean_broadcast = ops.broadcast_to(ops.reshape(mean, (1, self.dim)), x.shape)
            
            var = ops.summation((x - mean_broadcast) ** 2, axes=(0,)) / batch_size  # (dim,)
            
            # 更新 running 统计量（不参与计算图）
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean.data
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var.data
            
            # 标准化
            var_broadcast = ops.broadcast_to(ops.reshape(var, (1, self.dim)), x.shape)
            x_norm = (x - mean_broadcast) / ((var_broadcast + self.eps) ** 0.5)
        else:
            # 推理模式：使用 running 统计量
            mean_broadcast = ops.broadcast_to(ops.reshape(self.running_mean, (1, self.dim)), x.shape)
            var_broadcast = ops.broadcast_to(ops.reshape(self.running_var, (1, self.dim)), x.shape)
            x_norm = (x - mean_broadcast) / ((var_broadcast + self.eps) ** 0.5)
        
        # 应用仿射变换 gamma * x_norm + beta
        weight_broadcast = ops.broadcast_to(ops.reshape(self.weight, (1, self.dim)), x.shape)
        bias_broadcast = ops.broadcast_to(ops.reshape(self.bias, (1, self.dim)), x.shape)
        
        return weight_broadcast * x_norm + bias_broadcast
        ### END YOUR SOLUTION



class LayerNorm1d(Module):
    def __init__(self, dim: int, eps: float = 1e-5, device: Any | None = None, dtype: str = "float32") -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(
            init.ones(dim, device=device, dtype=dtype)
        )
        self.bias = Parameter(
            init.zeros(dim, device=device, dtype=dtype)
        )
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        # x: (batch_size, dim)
        batch_size = x.shape[0]
        
        # 沿特征维度（最后一维）计算均值
        mean = ops.summation(x, axes=(1,)) / self.dim  # (batch_size,)
        mean = ops.reshape(mean, (batch_size, 1))
        mean = ops.broadcast_to(mean, x.shape)
        
        # 沿特征维度计算方差
        var = ops.summation((x - mean) ** 2, axes=(1,)) / self.dim  # (batch_size,)
        var = ops.reshape(var, (batch_size, 1))
        var = ops.broadcast_to(var, x.shape)
        
        # 标准化
        x_norm = (x - mean) / ((var + self.eps) ** 0.5)
        
        # 应用仿射变换
        weight = ops.broadcast_to(ops.reshape(self.weight, (1, self.dim)), x.shape)
        bias = ops.broadcast_to(ops.reshape(self.bias, (1, self.dim)), x.shape)
        
        return weight * x_norm + bias
        ### END YOUR SOLUTION


class Dropout(Module):
    def __init__(self, p: float = 0.5) -> None:
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if self.training:
            # 生成 mask：以概率 (1-p) 保留
            mask = init.randb(*x.shape, p=1-self.p, device=x.device)
            # 缩放以保持期望值不变
            return x * mask / (1 - self.p)
        else:
            return x
        ### END YOUR SOLUTION


class Residual(Module):
    def __init__(self, fn: Module) -> None:
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return x + self.fn(x)
        ### END YOUR SOLUTION
