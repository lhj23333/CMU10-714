from typing import Optional, Any, Union
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from .ops_mathematic import *

import numpy as array_api

class LogSoftmax(TensorOp):
    def compute(self, Z: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        # LogSoftmax 沿最后一个轴计算
        # logsoftmax(z) = z - logsumexp(z)
        
        # 数值稳定：减去最大值
        max_z = array_api.max(Z, axis=-1, keepdims=True)
        z_shifted = Z - max_z
        
        # 计算 logsumexp（沿最后一个轴）
        exp_z = array_api.exp(z_shifted)
        sum_exp = array_api.sum(exp_z, axis=-1, keepdims=True)
        log_sum_exp = array_api.log(sum_exp)
        
        # logsoftmax = z - max - log(sum(exp(z - max))) = z_shifted - log_sum_exp
        return z_shifted - log_sum_exp
        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor):
        ### BEGIN YOUR SOLUTION
        # node 是 logsoftmax(Z) 的结果
        # softmax = exp(logsoftmax)
        softmax = exp(node)
        
        # 计算 sum(out_grad) 沿最后一个轴
        sum_out_grad = summation(out_grad, axes=(-1,))
        
        # 需要 reshape 和 broadcast 回原形状
        # sum_out_grad 的形状少了最后一维，需要加回来
        new_shape = list(node.shape)
        new_shape[-1] = 1
        sum_out_grad_reshaped = reshape(sum_out_grad, tuple(new_shape))
        sum_out_grad_broadcast = broadcast_to(sum_out_grad_reshaped, node.shape)
        
        # 梯度公式：grad = out_grad - softmax * sum(out_grad)
        return out_grad - softmax * sum_out_grad_broadcast
        ### END YOUR SOLUTION


def logsoftmax(a: Tensor) -> Tensor:
    return LogSoftmax()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None) -> None:
        self.axes = axes

    def compute(self, Z: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        # 沿指定轴找最大值（保持维度以便广播）
        max_z = array_api.max(Z, axis=self.axes, keepdims=True)
        
        # 数值稳定：减去最大值
        z_shifted = Z - max_z
        exp_z = array_api.exp(z_shifted)
        sum_exp = array_api.sum(exp_z, axis=self.axes, keepdims=True)
        
        # logsumexp = log(sum(exp(z-max))) + max
        result = array_api.log(sum_exp) + max_z
        
        # 去掉 keepdims 产生的多余维度
        if self.axes is not None:
            # 将指定轴 squeeze 掉
            result = array_api.squeeze(result, axis=self.axes)
        else:
            # axes=None 时结果是标量
            result = array_api.squeeze(result)
        
        return result
        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor):
        ### BEGIN YOUR SOLUTION
        Z = node.inputs[0]
        
        # node 是 logsumexp 的结果，形状已经被 reduce 了
        # 需要把它和 out_grad 广播回 Z 的形状
        
        if self.axes is not None:
            # 构建新形状：在被 reduce 的轴上插入 1
            new_shape = list(Z.shape)
            for axis in self.axes:
                new_shape[axis] = 1
            new_shape = tuple(new_shape)
            
            # reshape 然后 broadcast
            node_reshaped = reshape(node, new_shape)
            out_grad_reshaped = reshape(out_grad, new_shape)
            
            node_broadcast = broadcast_to(node_reshaped, Z.shape)
            out_grad_broadcast = broadcast_to(out_grad_reshaped, Z.shape)
        else:
            # axes=None，结果是标量
            node_broadcast = broadcast_to(reshape(node, (1,) * len(Z.shape)), Z.shape)
            out_grad_broadcast = broadcast_to(reshape(out_grad, (1,) * len(Z.shape)), Z.shape)
        
        # softmax = exp(Z - logsumexp(Z)) —— 数值稳定！
        softmax = exp(Z - node_broadcast)
        
        # 梯度 = out_grad * softmax
        return out_grad_broadcast * softmax
        ### END YOUR SOLUTIO


def logsumexp(a: Tensor, axes: Optional[tuple] = None) -> Tensor:
    return LogSumExp(axes=axes)(a)