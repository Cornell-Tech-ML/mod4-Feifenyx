from __future__ import annotations

from typing import TYPE_CHECKING

import minitorch

from . import operators
from .autodiff import Context

if TYPE_CHECKING:
    from typing import Tuple

    from .scalar import Scalar, ScalarLike


def wrap_tuple(x: float | Tuple[float, ...]) -> Tuple[float, ...]:
    """Turn a possible value into a tuple"""
    if isinstance(x, tuple):
        return x
    return (x,)


class ScalarFunction:
    """A wrapper for a mathematical function that processes and produces
    Scalar variables.

    This is a static class and is never instantiated. We use `class`
    here to group together the `forward` and `backward` code.
    """

    @classmethod
    def _backward(cls, ctx: Context, d_out: float) -> Tuple[float, ...]:
        return wrap_tuple(cls.backward(ctx, d_out))  # type: ignore

    @classmethod
    def _forward(cls, ctx: Context, *inps: float) -> float:
        return cls.forward(ctx, *inps)  # type: ignore

    @classmethod
    def apply(cls, *vals: ScalarLike) -> Scalar:
        """Applies the scalar function to the given values.

        Converts scalar-like inputs into `Scalar` objects and computes the
        output using the function's forward logic. Records the computational
        history for backpropagation.

        Args:
        ----
        cls: The class type of the ScalarFunction being applied.
        *vals: One or more scalar-like values (could be floats or Scalar objects)
               on which the scalar function is to be applied.

        Returns:
        -------
        Scalar: A `Scalar` object containing the result, with backpropagation
                history encapsulated in a `ScalarHistory` object.

        Raises:
        ------
        AssertionError: If the result is not a float.

        """
        raw_vals = []
        scalars = []
        for v in vals:
            if isinstance(v, minitorch.scalar.Scalar):
                scalars.append(v)
                raw_vals.append(v.data)
            else:
                scalars.append(minitorch.scalar.Scalar(v))
                raw_vals.append(float(v))

        # Create the context.
        ctx = Context(False)

        # Call forward with the variables.
        c = cls._forward(ctx, *raw_vals)
        assert isinstance(c, float), "Expected return type float got %s" % (type(c))

        # Create a new variable from the result with a new history.
        back = minitorch.scalar.ScalarHistory(cls, ctx, scalars)
        return minitorch.scalar.Scalar(c, back)


# Examples
class Add(ScalarFunction):
    """Addition function $f(x, y) = x + y$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Returns the sum of two values."""
        return a + b

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        """Returns the gradients of both inputs based on the backpropagated derivative."""
        return d_output, d_output


class Log(ScalarFunction):
    """Log function $f(x) = log(x)$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Computes the natural logarithm of the input."""
        ctx.save_for_backward(a)
        return operators.log(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Computes the gradient of the log function using the backpropagated derivative."""
        (a,) = ctx.saved_values
        return operators.log_back(a, d_output)


class Mul(ScalarFunction):
    """Multiplication function $f(x, y) = x * y$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Returns the product of two values."""
        ctx.save_for_backward(a, b)
        return a * b

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        """Returns the gradients of both inputs using the backpropagated derivative."""
        (a, b) = ctx.saved_values
        return d_output * b, d_output * a


class Inv(ScalarFunction):
    """Inverse function $f(x) = 1 / x$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Computes the inverse of the input."""
        ctx.save_for_backward(a)
        return 1.0 / a

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Computes the gradient of the inverse function using the backpropagated derivative."""
        (a,) = ctx.saved_values
        return operators.inv_back(a, d_output)


class Neg(ScalarFunction):
    """Negate function $f(x) = -x$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Returns the negation of the input."""
        return -a

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Returns the negated backpropagated derivative."""
        return -d_output


class Sigmoid(ScalarFunction):
    r"""Sigmoid function $f(x) = \frac{1.0}{(1.0 + e^{-x})}$ if $x >= 0$ else $\frac{e^x}{(1.0 + e^x)}$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Computes the sigmoid of the input."""
        ctx.save_for_backward(a)
        return operators.sigmoid(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Computes the gradient of the sigmoid function using the backpropagated derivative."""
        (a,) = ctx.saved_values
        return operators.sigmoid(a) * (1 - operators.sigmoid(a)) * d_output


class ReLU(ScalarFunction):
    """ReLU function $f(x) = x$ if $x > 0$ else 0"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Computes the ReLU activation of the input."""
        ctx.save_for_backward(a)
        return operators.relu(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Computes the gradient of the ReLU function using the backpropagated derivative."""
        (a,) = ctx.saved_values
        return operators.relu_back(a, d_output)


class Exp(ScalarFunction):
    """Exponential function $f(x) = e^x$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Computes the exponential of the input."""
        ctx.save_for_backward(a)
        return operators.exp(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Computes the gradient of the exponential function using the backpropagated derivative."""
        (a,) = ctx.saved_values
        return operators.exp(a) * d_output


class LT(ScalarFunction):
    """Less than function $f(x, y) = 1.0$ if $x < y$ else $0.0$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Returns 1.0 if `a` is less than `b`, else 0.0."""
        return 1.0 if a < b else 0.0

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        """Returns zero gradients since the forward pass output is constant (1.0 or 0.0)."""
        return 0.0, 0.0


class EQ(ScalarFunction):
    """Equality function $f(x, y) = 1.0$ if $x == y$ else $0.0$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Returns 1.0 if `a` is equal to `b`, else 0.0."""
        return 1.0 if a == b else 0.0

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        """Returns zero gradients since the forward pass output is constant (1.0 or 0.0)."""
        return 0.0, 0.0
