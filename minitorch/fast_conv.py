from typing import Tuple, TypeVar, Any

from numba import prange
from numba import njit as _njit

from .autodiff import Context
from .tensor import Tensor
from .tensor_data import (
    Shape,
    Strides,
    Storage,
    broadcast_index,
    index_to_position,
    to_index,
)
from .tensor_functions import Function

Fn = TypeVar("Fn")


def njit(fn: Fn, **kwargs: Any) -> Fn:
    """Just-in-time compile the given function with Numba."""
    return _njit(inline="always", **kwargs)(fn)  # type: ignore


# This code will JIT compile fast versions your tensor_data functions.
# If you get an error, read the docs for NUMBA as to what is allowed
# in these functions.
to_index = njit(to_index)
index_to_position = njit(index_to_position)
broadcast_index = njit(broadcast_index)


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
    """1D Convolution implementation.

    Given input tensor of

       `batch, in_channels, width`

    and weight tensor

       `out_channels, in_channels, k_width`

    Computes padded output of

       `batch, out_channels, width`

    `reverse` decides if weight is anchored left (False) or right.
    (See diagrams)

    Args:
    ----
        out (Storage): storage for `out` tensor.
        out_shape (Shape): shape for `out` tensor.
        out_strides (Strides): strides for `out` tensor.
        out_size (int): size of the `out` tensor.
        input (Storage): storage for `input` tensor.
        input_shape (Shape): shape for `input` tensor.
        input_strides (Strides): strides for `input` tensor.
        weight (Storage): storage for `input` tensor.
        weight_shape (Shape): shape for `input` tensor.
        weight_strides (Strides): strides for `input` tensor.
        reverse (bool): anchor weight at left or right

    """
    batch_, out_channels, out_width = out_shape
    batch, in_channels, width = input_shape
    out_channels_, in_channels_, kw = weight_shape

    assert (
        batch == batch_
        and in_channels == in_channels_
        and out_channels == out_channels_
    )
    s1 = input_strides
    s2 = weight_strides

    for b in prange(batch):
        for oc in prange(out_channels):
            for ow in prange(out_width):
                out_position = (
                    b * out_strides[0] + oc * out_strides[1] + ow * out_strides[2]
                )
                conv_sum = 0.0

                # Iterate over input channels and kernel width
                for ic in range(in_channels):
                    for kw_pos in range(kw):
                        if reverse:
                            # For reverse convolution, slide window right to left
                            iw = ow - kw_pos
                            # Flip the kernel for reverse convolution
                            weight_position = (
                                oc * s2[0] + ic * s2[1] + (kw - 1 - kw_pos) * s2[2]
                            )
                        else:
                            # For forward convolution, slide window left to right
                            iw = ow + kw_pos
                            # Use normal kernel order for forward convolution
                            weight_position = oc * s2[0] + ic * s2[1] + kw_pos * s2[2]

                        # Check if input width is within bounds
                        if 0 <= iw < width:
                            input_position = b * s1[0] + ic * s1[1] + iw * s1[2]
                            # Perform convolution operation and accumulate result
                            conv_sum += input[input_position] * weight[weight_position]
                out[out_position] = conv_sum


tensor_conv1d = njit(_tensor_conv1d, parallel=True)


class Conv1dFun(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, weight: Tensor) -> Tensor:
        """Compute a 1D Convolution

        Args:
        ----
            ctx : Context
            input : batch x in_channel x h x w
            weight : out_channel x in_channel x kh x kw

        Returns:
        -------
            batch x out_channel x h x w

        """
        ctx.save_for_backward(input, weight)
        batch, in_channels, w = input.shape
        out_channels, in_channels2, kw = weight.shape
        assert in_channels == in_channels2

        # Run convolution
        output = input.zeros((batch, out_channels, w))
        tensor_conv1d(
            *output.tuple(), output.size, *input.tuple(), *weight.tuple(), False
        )
        return output

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Compute gradients for 1D Convolution

        Args:
        ----
            ctx (Context): Context containing saved tensors from forward pass
            grad_output (Tensor): Gradient of the loss with respect to the output

        Returns:
        -------
            Tuple[Tensor, Tensor]: Gradients with respect to input and weight

        """
        input, weight = ctx.saved_values
        batch, in_channels, w = input.shape
        out_channels, in_channels, kw = weight.shape
        grad_weight = grad_output.zeros((in_channels, out_channels, kw))
        new_input = input.permute(1, 0, 2)
        new_grad_output = grad_output.permute(1, 0, 2)
        tensor_conv1d(  # type: ignore
            *grad_weight.tuple(),
            grad_weight.size,
            *new_input.tuple(),
            *new_grad_output.tuple(),
            False,  # type: ignore
        )
        grad_weight = grad_weight.permute(1, 0, 2)

        grad_input = input.zeros((batch, in_channels, w))
        new_weight = weight.permute(1, 0, 2)
        tensor_conv1d(  # type: ignore
            *grad_input.tuple(),
            grad_input.size,  # type: ignore
            *grad_output.tuple(),
            *new_weight.tuple(),
            True,  # type: ignore
        )
        return grad_input, grad_weight


conv1d = Conv1dFun.apply


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
    """2D Convolution implementation.

    Given input tensor of

       `batch, in_channels, height, width`

    and weight tensor

       `out_channels, in_channels, k_height, k_width`

    Computes padded output of

       `batch, out_channels, height, width`

    `Reverse` decides if weight is anchored top-left (False) or bottom-right.
    (See diagrams)


    Args:
    ----
        out (Storage): storage for `out` tensor.
        out_shape (Shape): shape for `out` tensor.
        out_strides (Strides): strides for `out` tensor.
        out_size (int): size of the `out` tensor.
        input (Storage): storage for `input` tensor.
        input_shape (Shape): shape for `input` tensor.
        input_strides (Strides): strides for `input` tensor.
        weight (Storage): storage for `input` tensor.
        weight_shape (Shape): shape for `input` tensor.
        weight_strides (Strides): strides for `input` tensor.
        reverse (bool): anchor weight at top-left or bottom-right

    """
    batch_, out_channels, out_height, out_width = out_shape
    batch, in_channels, height, width = input_shape
    out_channels_, in_channels_, kh, kw = weight_shape

    assert (
        batch == batch_
        and in_channels == in_channels_
        and out_channels == out_channels_
    )

    s1 = input_strides
    s2 = weight_strides
    # inners
    s10, s11, s12, s13 = s1[0], s1[1], s1[2], s1[3]
    s20, s21, s22, s23 = s2[0], s2[1], s2[2], s2[3]

    for b in prange(batch):
        for oc in prange(out_channels):
            for oh in prange(out_height):
                for ow in prange(out_width):
                    out_position = (
                        b * out_strides[0]
                        + oc * out_strides[1]
                        + oh * out_strides[2]
                        + ow * out_strides[3]
                    )
                    conv_sum = 0.0

                    # Iterate over input channels and kernel dimensions
                    for ic in range(in_channels):
                        for kh_pos in range(kh):
                            for kw_pos in range(kw):
                                if reverse:
                                    # For reverse convolution, slide window bottom-right to top-left
                                    ih = oh - kh_pos
                                    iw = ow - kw_pos
                                    # Flip the kernel for reverse convolution
                                    weight_position = (
                                        oc * s20
                                        + ic * s21
                                        + (kh - 1 - kh_pos) * s22
                                        + (kw - 1 - kw_pos) * s23
                                    )
                                else:
                                    # For forward convolution, slide window top-left to bottom-right
                                    ih = oh + kh_pos
                                    iw = ow + kw_pos
                                    # Use normal kernel order for forward convolution
                                    weight_position = (
                                        oc * s20
                                        + ic * s21
                                        + kh_pos * s22
                                        + kw_pos * s23
                                    )

                                # Check if input position is within bounds
                                if 0 <= ih < height and 0 <= iw < width:
                                    input_position = (
                                        b * s10 + ic * s11 + ih * s12 + iw * s13
                                    )
                                    # Perform convolution operation and accumulate result
                                    conv_sum += (
                                        input[input_position] * weight[weight_position]
                                    )

                    out[out_position] = conv_sum


tensor_conv2d = njit(_tensor_conv2d, parallel=True, fastmath=True)


class Conv2dFun(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, weight: Tensor) -> Tensor:
        """Compute a 2D Convolution

        Args:
        ----
            ctx : Context
            input : batch x in_channel x h x w
            weight  : out_channel x in_channel x kh x kw

        Returns:
        -------
            (:class:`Tensor`) : batch x out_channel x h x w

        """
        ctx.save_for_backward(input, weight)
        batch, in_channels, h, w = input.shape
        out_channels, in_channels2, kh, kw = weight.shape
        assert in_channels == in_channels2
        output = input.zeros((batch, out_channels, h, w))
        tensor_conv2d(
            *output.tuple(), output.size, *input.tuple(), *weight.tuple(), False
        )
        return output

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Compute gradients for 2D Convolution

        Args:
        ----
            ctx (Context): Context containing saved tensors from forward pass
            grad_output (Tensor): Gradient of the loss with respect to the output

        Returns:
        -------
            Tuple[Tensor, Tensor]: Gradients with respect to input and weight

        """
        input, weight = ctx.saved_values
        batch, in_channels, h, w = input.shape
        out_channels, in_channels, kh, kw = weight.shape

        grad_weight = grad_output.zeros((in_channels, out_channels, kh, kw))
        new_input = input.permute(1, 0, 2, 3)
        new_grad_output = grad_output.permute(1, 0, 2, 3)
        tensor_conv2d(  # type: ignore
            *grad_weight.tuple(),
            grad_weight.size,
            *new_input.tuple(),
            *new_grad_output.tuple(),
            False,  # type: ignore
        )
        grad_weight = grad_weight.permute(1, 0, 2, 3)

        grad_input = input.zeros((batch, in_channels, h, w))
        new_weight = weight.permute(1, 0, 2, 3)
        tensor_conv2d(  # type: ignore
            *grad_input.tuple(),
            grad_input.size,  # type: ignore
            *grad_output.tuple(),
            *new_weight.tuple(),
            True,  # type: ignore
        )
        return grad_input, grad_weight


conv2d = Conv2dFun.apply
