from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Tuple, Protocol


# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
    ----
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
    -------
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$

    """
    vals_plus = vals[:arg] + (vals[arg] + epsilon,) + vals[arg + 1 :]
    vals_minus = vals[:arg] + (vals[arg] - epsilon,) + vals[arg + 1 :]

    return (f(*vals_plus) - f(*vals_minus)) / (2.0 * epsilon)


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        """Add `val` to the the derivative accumulated on this variable.
        Should only be called during autodifferentiation on leaf variables.

        Args:
        ----
            x: value to be accumulated

        """
        ...

    @property
    def unique_id(self) -> int:
        """Gets the unique identifier for the object"""
        ...

    def is_leaf(self) -> bool:
        """True if this variable created by the user"""
        ...

    def is_constant(self) -> bool:
        """True if this object is a constant value"""
        ...

    @property
    def parents(self) -> Iterable["Variable"]:
        """Retrieves the parent variables of the object"""
        ...

    def chain_rule(self, d_output: Any) -> Iterable[Tuple[Variable, Any]]:
        """Computes the gradients for the parent variables using the chain rule."""
        ...


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """Computes the topological order of the computation graph.

    Args:
    ----
        variable: The right-most variable

    Returns:
    -------
        Non-constant Variables in topological order starting from the right.

    """
    visited = set()
    order = []

    def DFS(var: Variable) -> None:
        """Depth-first search helper function."""
        if var.unique_id in visited:
            return
        visited.add(var.unique_id)

        if not var.is_leaf():
            for parent in var.parents:
                if not parent.is_constant():
                    DFS(parent)

        if not var.is_constant():
            order.insert(0, var)

    DFS(variable)

    return order


def backpropagate(variable: Variable, deriv: Any) -> None:
    """Runs backpropagation on the computation graph in order to
    compute derivatives for the leaf nodes.

    Args:
    ----
    variable: The right-most variable
    deriv  : Its derivative that we want to propagate backward to the leaves.

    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.

    """
    order = topological_sort(variable)

    derivative = {variable.unique_id: deriv}

    for var in order:
        if var.is_leaf():
            var.accumulate_derivative(derivative[var.unique_id])

        else:
            for v, d_output in var.chain_rule(derivative.get(var.unique_id, 0.0)):
                derivative[v.unique_id] = derivative.get(v.unique_id, 0.0) + d_output


@dataclass
class Context:
    """Context class is used by `Function` to store information during the forward pass."""

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        """Store the given `values` if they need to be used during backpropagation."""
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        """Retrieves the values stored for use during backpropagation."""
        return self.saved_values
