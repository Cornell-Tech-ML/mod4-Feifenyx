# type: ignore
# Currently pyright doesn't support numba.cuda

from typing import Tuple, TypeVar, Any

import numba
from numba import cuda
from numba.cuda import jit as _jit
from .autodiff import Context
from .tensor import Tensor
from .tensor_data import (
    MAX_DIMS,
    Index,
    Shape,
    Strides,
    Storage,
    broadcast_index,
    index_to_position,
    to_index,
)
from .tensor_functions import Function

FakeCUDAKernel = Any

# This code will CUDA compile fast versions your tensor_data functions.
# If you get an error, read the docs for NUMBA as to what is allowed
# in these functions.

Fn = TypeVar("Fn")

def device_jit(fn: Fn, **kwargs: Any) -> Fn:
    """Decorator for JIT compilation of device functions."""
    return _jit(device=True, **kwargs)(fn)  # type: ignore


def jit(fn: Fn, **kwargs: Any) -> FakeCUDAKernel:
    """Decorator for JIT compilation of CUDA kernels."""
    return _jit(**kwargs)(fn)  # type: ignore

to_index = device_jit(to_index)
index_to_position = device_jit(index_to_position)
broadcast_index = device_jit(broadcast_index)

THREADS_PER_BLOCK = 32

def _tensor_conv1d(
    out: Storage,
    out_shape: Shape,
    out_strides: Strides,
    out_size: int,
    input: Storage,
    input_shape: Shape,
    input_strides: Strides,
    weight: Storage,
    weight_shape: Shape,
    weight_strides: Strides,
    reverse: bool,
) -> None:
    batch_, out_channels, out_width = out_shape
    batch, in_channels, width = input_shape
    out_channels_, in_channels_, kw = weight_shape

    s1 = input_strides
    s2 = weight_strides

    # Use shared memory for input and weight
    BLOCK_DIM = 32
    shared_input = cuda.shared.array(shape=(BLOCK_DIM, BLOCK_DIM), dtype=numba.float32)
    shared_weight = cuda.shared.array(shape=(BLOCK_DIM, BLOCK_DIM), dtype=numba.float32)

    # Get thread indices
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    bx = cuda.blockIdx.x
    by = cuda.blockIdx.y

    # Calculate global indices
    b = bx * BLOCK_DIM + tx
    oc = by * BLOCK_DIM + ty

    if b < batch and oc < out_channels:
        for ow in range(out_width):
            conv_sum = 0.0
            for ic in range(0, in_channels, BLOCK_DIM):
                # Load input and weight into shared memory
                if ic + ty < in_channels and ow + tx < width:
                    shared_input[ty, tx] = input[b * s1[0] + (ic + ty) * s1[1] + (ow + tx) * s1[2]]
                if ic + tx < in_channels and ty < kw:
                    shared_weight[tx, ty] = weight[oc * s2[0] + (ic + tx) * s2[1] + ty * s2[2]]
                cuda.syncthreads()

                # Compute convolution
                for k in range(min(BLOCK_DIM, kw)):
                    for j in range(min(BLOCK_DIM, in_channels - ic)):
                        if reverse:
                            iw = ow - k
                        else:
                            iw = ow + k
                        if 0 <= iw < width:
                            conv_sum += shared_input[j, tx] * shared_weight[j, k]
                cuda.syncthreads()

            # Write result to output
            out_position = b * out_strides[0] + oc * out_strides[1] + ow * out_strides[2]
            out[out_position] = conv_sum


tensor_conv1d = cuda.jit()(_tensor_conv1d)


class Conv1dFunCuda(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, weight: Tensor) -> Tensor:
        ctx.save_for_backward(input, weight)
        batch, in_channels, w = input.shape
        out_channels, in_channels2, kw = weight.shape
        assert in_channels == in_channels2
        
        output_width = w - kw + 1
        output = input.zeros((batch, out_channels, output_width))

        blockspergrid = (batch, out_channels)
        threadsperblock = THREADS_PER_BLOCK

        tensor_conv1d[blockspergrid, threadsperblock](
            *output.tuple(), output.size,
            *input.tuple(), *weight.tuple(), False
        )
        
        return output

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        input, weight = ctx.saved_values
        batch, in_channels, w = input.shape
        out_channels, in_channels, kw = weight.shape
        
        grad_input = input.zeros((batch, in_channels, w))
        grad_weight = weight.zeros((out_channels, in_channels, kw))
        
        blockspergrid = (grad_output.shape[0], grad_output.shape[1])
        threadsperblock = THREADS_PER_BLOCK

        # Compute grad_weight
        new_input = input.permute(1, 0, 2)
        new_grad_output = grad_output.permute(1, 0, 2)
        tensor_conv1d[blockspergrid, threadsperblock](
            *grad_weight.tuple(), grad_weight.size,
            *new_input.tuple(), *new_grad_output.tuple(), False
        )
        grad_weight = grad_weight.permute(1, 0, 2)
        
        # Compute grad_input
        new_weight = weight.permute(1, 0, 2)
        tensor_conv1d[blockspergrid, threadsperblock](
            *grad_input.tuple(), grad_input.size,
            *grad_output.tuple(), *new_weight.tuple(), True
        )
        
        return grad_input, grad_weight


conv1d_cuda = Conv1dFunCuda.apply


def _tensor_conv2d(
    out: Storage,
    out_shape: Shape,
    out_strides: Strides,
    out_size: int,
    input: Storage,
    input_shape: Shape,
    input_strides: Strides,
    weight: Storage,
    weight_shape: Shape,
    weight_strides: Strides,
    reverse: bool,
) -> None:
    batch_, out_channels, out_height, out_width = out_shape
    batch, in_channels, height, width = input_shape
    out_channels_, in_channels_, kh, kw = weight_shape

    s1 = input_strides
    s2 = weight_strides

    # Use shared memory for input and weight
    BLOCK_DIM = 32
    shared_input = cuda.shared.array(shape=(BLOCK_DIM, BLOCK_DIM), dtype=numba.float32)
    shared_weight = cuda.shared.array(shape=(BLOCK_DIM, BLOCK_DIM), dtype=numba.float32)

    # Get thread indices
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y

    # Calculate global indices
    b = cuda.blockIdx.z
    oc = cuda.blockIdx.y * BLOCK_DIM + cuda.threadIdx.y
    oh = cuda.blockIdx.x * BLOCK_DIM + cuda.threadIdx.x

    if b < batch and oc < out_channels:
        for ow in range(out_width):
            conv_sum = 0.0
            for ic in range(0, in_channels, BLOCK_DIM):
                for kh_pos in range(0, kh, BLOCK_DIM):
                    # Load input and weight into shared memory
                    if ic + tx < in_channels and oh + ty < height:
                        shared_input[tx, ty] = input[b * s1[0] + (ic + tx) * s1[1] + (oh + ty) * s1[2] + ow * s1[3]]
                    if ic + tx < in_channels and kh_pos + ty < kh:
                        shared_weight[tx, ty] = weight[oc * s2[0] + (ic + tx) * s2[1] + (kh_pos + ty) * s2[2]]
                    cuda.syncthreads()

                    # Compute convolution
                    for k in range(min(BLOCK_DIM, kw)):
                        for j in range(min(BLOCK_DIM, in_channels - ic)):
                            for i in range(min(BLOCK_DIM, kh - kh_pos)):
                                if reverse:
                                    ih = oh - kh_pos - i
                                    iw = ow - k
                                else:
                                    ih = oh + kh_pos + i
                                    iw = ow + k
                                if 0 <= ih < height and 0 <= iw < width:
                                    conv_sum += shared_input[j, i] * shared_weight[j, i]
                    cuda.syncthreads()

            # Write result to output
            out_position = b * out_strides[0] + oc * out_strides[1] + oh * out_strides[2] + ow * out_strides[3]
            out[out_position] = conv_sum


tensor_conv2d = cuda.jit()(_tensor_conv2d)


class Conv2dFunCuda(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, weight: Tensor) -> Tensor:
        ctx.save_for_backward(input, weight)
        batch, in_channels, h, w = input.shape
        out_channels, in_channels2, kh, kw = weight.shape
        assert in_channels == in_channels2
        
        output_height = h - kh + 1
        output_width = w - kw + 1
        output = input.zeros((batch, out_channels, output_height, output_width))

        blockspergrid = (
            (output_height + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK,
            (output_width + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK,
            batch
        )
        threadsperblock = (THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1)

        tensor_conv2d[blockspergrid, threadsperblock](
            *output.tuple(), output.size,
            *input.tuple(), *weight.tuple(), False
        )
        
        return output

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        input, weight = ctx.saved_values
        batch, in_channels, h, w = input.shape
        out_channels, in_channels, kh, kw = weight.shape
        
        grad_input = input.zeros((batch, in_channels, h, w))
        grad_weight = weight.zeros((out_channels, in_channels, kh, kw))
        
        blockspergrid = (
            (h + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK,
            (w + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK,
            batch
        )
        threadsperblock = (THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1)
        
        # Compute grad_weight
        new_input = input.permute(1, 0, 2, 3)
        new_grad_output = grad_output.permute(1, 0, 2, 3)
        tensor_conv2d[blockspergrid, threadsperblock](
            *grad_weight.tuple(), grad_weight.size,
            *new_input.tuple(), *new_grad_output.tuple(), False
        )
        grad_weight = grad_weight.permute(1, 0, 2, 3)
        
        # Compute grad_input
        new_weight = weight.permute(1, 0, 2, 3)
        tensor_conv2d[blockspergrid, threadsperblock](
            *grad_input.tuple(), grad_input.size,
            *grad_output.tuple(), *new_weight.tuple(), True
        )
        
        return grad_input, grad_weight


conv2d_cuda = Conv2dFunCuda.apply