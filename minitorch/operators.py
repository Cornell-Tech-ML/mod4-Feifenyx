"""Collection of the core mathematical operators used throughout the code base."""

import math

# ## Task 0.1
from typing import Callable, Iterable

#
# Implementation of a prelude of elementary functions.

# Mathematical functions:
# - mul
# - id
# - add
# - neg
# - lt
# - eq
# - max
# - is_close
# - sigmoid
# - relu
# - log
# - exp
# - log_back
# - inv
# - inv_back
# - relu_back
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$


def mul(x: float, y: float) -> float:
    """Multiply `x` by `y`"""
    return x * y


def id(x: float) -> float:
    """Return `x` unchanged"""
    return x


def add(x: float, y: float) -> float:
    """Add `x` to `y`"""
    return x + y


def neg(x: float) -> float:
    """Negate `x`"""
    return -x


def lt(x: float, y: float) -> float:
    """Check if `x` is less than `y`"""
    return 1.0 if x < y else 0.0


def eq(x: float, y: float) -> float:
    """Check if `x` is equal to `y`"""
    return 1.0 if x == y else 0.0


def max(x: float, y: float) -> float:
    """Return the larger of `x` and `y`"""
    return x if x > y else y


def is_close(x: float, y: float) -> float:
    """Check if the difference between `x` and `y` is less than 0.01"""
    return 1.0 if abs(x - y) < 1e-2 else 0.0


def sigmoid(x: float) -> float:
    """Calculate the sigmoid function of `x`"""
    return 1.0 / (1.0 + math.exp(-x)) if x >= 0 else math.exp(x) / (1.0 + math.exp(x))


def relu(x: float) -> float:
    """Return `x` if `x` is greater than 0, else 0"""
    return x if x > 0 else 0.0


def log(x: float) -> float:
    """Calculate the natural logarithm of `x`"""
    return math.log(x)


def exp(x: float) -> float:
    """Calculate the exponential function of `x`"""
    return math.exp(x)


def inv(x: float) -> float:
    """Calculate the reciprocal of `x`"""
    return 1.0 / x


def log_back(x: float, y: float) -> float:
    """Compute the derivative of log `x` times `y`"""
    return y / x


def inv_back(x: float, y: float) -> float:
    """Compute the derivative of reciprocal `x` times `y`"""
    return -y / x**2


def relu_back(x: float, y: float) -> float:
    """Compute the derivative of ReLU `x` times `y`"""
    return y if x > 0 else 0.0


# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists


def map(fn: Callable[[float], float], ls: Iterable[float]) -> Iterable[float]:
    """Apply the function `fn` to each element of the iterable `ls`"""
    return [fn(x) for x in ls]


def zipWith(
    fn: Callable[[float, float], float], ls1: Iterable[float], ls2: Iterable[float]
) -> Iterable[float]:
    """Combine elements from the iterables `ls1` and `ls2` using the function `fn`"""
    ls1, ls2, ret = list(ls1), list(ls2), []
    for i in range(len(ls1)):
        ret.append(fn(ls1[i], ls2[i]))
    return ret


def reduce(fn: Callable[[float, float], float], ls: Iterable[float]) -> float:
    """Reduce the iterable `ls` to a single value using the function `fn`"""
    if not ls:
        return 0

    ls = list(ls)
    i, ret = 1, ls[0]
    while i < len(ls):
        ret = fn(ret, ls[i])
        i += 1
    return ret


def negList(ls: Iterable[float]) -> Iterable[float]:
    """Negate all elements in a list `ls`"""
    return map(neg, ls)


def addLists(ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
    """Add corresponding elements from the lists `ls1` and `ls2`"""
    return zipWith(add, ls1, ls2)


def sum(ls: Iterable[float]) -> float:
    """Sum all elements in the list `ls`"""
    return reduce(add, ls)


def prod(ls: Iterable[float]) -> float:
    """Calculate the product of all elements in the list `ls`"""
    return reduce(mul, ls)
