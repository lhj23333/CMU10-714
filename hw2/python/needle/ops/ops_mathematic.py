"""Operator implementations."""

from numbers import Number
from typing import Optional, List, Tuple, Union

from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp
import numpy

# NOTE: we will import numpy as the array_api
# as the backend for our computations, this line will change in later homeworks

BACKEND = "np"
import numpy as array_api

class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class EWisePow(TensorOp):
    """Op to element-wise raise a tensor to a power."""

    def compute(self, a: NDArray, b: NDArray) -> NDArray:
        return array_api.power(a, b)
        
    def gradient(self, out_grad, node):
        a, b = node.inputs

        # 对a的梯度：d/dx (a^b) = b * a^(b-1) 
        grad_a = out_grad * b * power(a, b - 1)

        # 对b的梯度 d/dx (a^b) = a^b * ln(a)
        grad_b = out_grad * power(a, b) * log(a)

        return grad_a, grad_b

def power(a, b):
    return EWisePow()(a, b)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        return array_api.power(a, self.scalar)

    def gradient(self, out_grad, node):
        # 获取输入张量
        a = node.inputs[0]

        # 对a的梯度：out_grad * n * a^(n-1)
        grad_a = out_grad * self.scalar * power_scalar(a, self.scalar - 1)

        return (grad_a, )


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        return a / b

    def gradient(self, out_grad, node):
        a, b = node.inputs

        # 对a的梯度 d/dx (a/b) = 1/b
        grad_a = out_grad / b
        # 对b的梯度 d/dx (a/b) = -a/(b^2)
        grad_b = out_grad * (-a / (b * b))

        return grad_a, grad_b

def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        return a / self.scalar

    def gradient(self, out_grad, node):
        return (out_grad * (1 / self.scalar), )


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        if self.axes:
            return array_api.swapaxes(a, self.axes[0], self.axes[1])
        else:
            return array_api.swapaxes(a, a.ndim - 2, a.ndim - 1)

    def gradient(self, out_grad, node):
        return out_grad.transpose(self.axes)


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        return array_api.reshape(a, self.shape)

    def gradient(self, out_grad, node):
        # 获取输入张量的形状
        input_shape = node.inputs[0].shape
        # 将输出梯度重新调整为输入张量的形状
        return (reshape(out_grad, input_shape),)


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        return array_api.broadcast_to(a, self.shape)      

    def gradient(self, out_grad, node):
        input_shape = node.inputs[0].shape
        output_shape = self.shape
        
        # 找出需要求和的轴
        axes = []
        
        # 处理维度数量不同的情况
        ndim_diff = len(output_shape) - len(input_shape)
        for i in range(ndim_diff):
            axes.append(i)
        
        # 被广播的维度（原来是1的维度）需要求和
        for i, (input_dim, output_dim) in enumerate(zip(input_shape, output_shape[ndim_diff:])):
            if input_dim == 1 and output_dim > 1:
                axes.append(i + ndim_diff)
        
        # 沿着这些轴求和
        if axes:
            grad = summation(out_grad, axes=tuple(axes))
        else:
            grad = out_grad
        
        # 重塑回输入形状
        return (reshape(grad, input_shape),)

def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        return array_api.sum(a, axis=self.axes)

    def gradient(self, out_grad, node):
        input_shape = node.inputs[0].shape
        
        if self.axes is None:
            # 对所有元素求和，广播回原形状
            return (broadcast_to(reshape(out_grad, (1,) * len(input_shape)), input_shape),)
        else:
            # 处理单个 int 的情况
            axes = self.axes if isinstance(self.axes, tuple) else (self.axes,)
            
            # 构建新形状，在求和的维度插入 1
            new_shape = list(input_shape)
            for axis in axes:
                new_shape[axis] = 1
            
            # 重塑并广播
            return (broadcast_to(reshape(out_grad, tuple(new_shape)), input_shape),)

def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
         return array_api.matmul(a, b)

    def gradient(self, out_grad, node):
        lhs, rhs = node.inputs
        lgrad = matmul(out_grad, rhs.transpose())
        rgrad = matmul(lhs.transpose(), out_grad)

        # 调整梯度形状以匹配输入张量形状
        if len(lhs.shape) < len(lgrad.shape):
            lgrad = lgrad.sum(tuple([i for i in range(len(lgrad.shape) - len(lhs.shape))]))
        if len(rhs.shape) < len(rgrad.shape):
            rgrad = rgrad.sum(tuple([i for i in range(len(rgrad.shape) - len(rhs.shape))]))
        
        return lgrad, rgrad


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        return -a

    def gradient(self, out_grad, node):
        return (negate(out_grad),) 


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        return array_api.log(a)

    def gradient(self, out_grad, node):
        # 获取输入张量
        a = node.inputs[0]

        # 对a的梯度：d/dx (log(a)) = 1/a
        return (out_grad / a,)


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        return array_api.exp(a)


    def gradient(self, out_grad, node):
        # 获取输入张量
        a = node.inputs[0]

        # 对a的梯度：d/dx (exp(a)) = exp(a)
        return (out_grad * exp(a),)


def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a):
        return array_api.maximum(a, 0)

    def gradient(self, out_grad, node):
        a = node.inputs[0]
        mask = Tensor(a.realize_cached_data() > 0)
        return (out_grad * mask,) 


def relu(a):
    return ReLU()(a)

