from typing import Tuple

from . import operators
from .autodiff import Context
from .fast_ops import FastOps
from .tensor import Tensor
from .tensor_functions import Function, rand


# List of functions in this file:
# - avgpool2d: Tiled average pooling 2D
# - argmax: Compute the argmax as a 1-hot tensor
# - Max: New Function for max operator
# - max: Apply max reduction
# - softmax: Compute the softmax as a tensor
# - logsoftmax: Compute the log of the softmax as a tensor - See https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations
# - maxpool2d: Tiled max pooling 2D
# - dropout: Dropout positions based on random noise, include an argument to turn off


def tile(input: Tensor, kernel: Tuple[int, int]) -> Tuple[Tensor, int, int]:
    """Reshape an image tensor for 2D pooling

    Args:
    ----
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
    -------
        Tensor of size batch x channel x new_height x new_width x (kernel_height * kernel_width) as well as the new_height and new_width value.

    """
    batch, channel, height, width = input.shape
    kh, kw = kernel
    assert height % kh == 0
    assert width % kw == 0

    new_height = height // kh
    new_width = width // kw

    # Reshape the input tensor
    reshaped = input.contiguous().view(batch, channel, new_height, kh, new_width, kw)

    # Permute the dimensions to group the kernel elements
    permuted = reshaped.permute(0, 1, 2, 4, 3, 5)

    # Reshape to combine the kernel dimensions
    tiled = permuted.contiguous().view(batch, channel, new_height, new_width, kh * kw)

    return tiled, new_height, new_width


def avgpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Perform 2D average pooling on the input tensor

    Args:
    ----
        input (Tensor): Input tensor of shape (batch, channel, height, width)
        kernel (Tuple[int, int]): Pooling kernel size (height, width)

    Returns:
    -------
        Tensor: Pooled tensor

    """
    batch, channel, _, _ = input.shape
    tiled, new_height, new_width = tile(input, kernel)
    pooled = tiled.mean(dim=-1)
    return pooled.view(batch, channel, new_height, new_width)


max_reduce = FastOps.reduce(operators.max, -1e9)


def argmax(input: Tensor, dim: int) -> Tensor:
    """Compute the indices of the maximum values along a dimension"""
    out = max_reduce(input, dim)
    return out == input


class Max(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, dim: Tensor) -> Tensor:
        """Compute the maximum values along a dimension"""
        ctx.save_for_backward(input, int(dim.item()))
        return max_reduce(input, int(dim.item()))

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        """Compute gradients for the max operation"""
        input, dim = ctx.saved_values
        return grad_output.f.mul_zip(argmax(input, dim), grad_output), 0.0


def max(input: Tensor, dim: int) -> Tensor:
    """Apply the Max operation along a given dimension."""
    return Max.apply(input, input._ensure_tensor(dim))


def softmax(input: Tensor, dim: int) -> Tensor:
    """Compute the softmax of the input tensor along a dimension"""
    input_exp = input.exp()
    return input_exp / input_exp.sum(dim=dim)


def logsoftmax(input: Tensor, dim: int) -> Tensor:
    """Compute the log softmax of the input tensor along a dimension.

    This function uses the log-sum-exp trick for numerical stability.
    See https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations

    Args:
    ----
        input (Tensor): Input tensor
        dim (int): Dimension along which to compute log softmax

    Returns:
    -------
        Tensor: Log softmax of the input tensor

    """
    input_max = max(input, dim=dim)
    input_shifted = input - input_max
    return input_shifted - input_shifted.exp().sum(dim=dim).log()


def maxpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Perform 2D max pooling on the input tensor

    Args:
    ----
        input (Tensor): Input tensor of shape (batch, channel, height, width)
        kernel (Tuple[int, int]): Pooling kernel size (height, width)

    Returns:
    -------
        Tensor: Pooled tensor

    """
    batch, channel, _, _ = input.shape
    tiled, new_height, new_width = tile(input, kernel)
    pooled = max(tiled, dim=4)
    return pooled.view(batch, channel, new_height, new_width)


def dropout(input: Tensor, p: float = 0.5, ignore: bool = False) -> Tensor:
    """Apply dropout to the input tensor

    Args:
    ----
        input (Tensor): Input tensor
        p (float, optional): Probability of an element to be zeroed. Default: 0.5
        ignore (bool, optional): If True, don't apply dropout. Default: False

    Returns:
    -------
        Tensor: Tensor with dropout applied

    """
    if ignore or p == 0:
        return input
    return input * (rand(input.shape) > p)
