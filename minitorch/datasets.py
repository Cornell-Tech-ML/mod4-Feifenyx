import math
import random
from dataclasses import dataclass
from typing import List, Tuple


def make_pts(N: int) -> List[Tuple[float, float]]:
    """Generates a list of N random 2D points.

    Args:
    ----
        N (int): The number of points to generate.

    Returns:
    -------
        List[Tuple[float, float]]: A list containing N tuples, each representing
                                   a 2D point with two random float coordinates
                                   (x_1, x_2) where 0 <= x_1, x_2 < 1.

    """
    X = []
    for i in range(N):
        x_1 = random.random()
        x_2 = random.random()
        X.append((x_1, x_2))
    return X


@dataclass
class Graph:
    N: int
    X: List[Tuple[float, float]]
    y: List[int]


def simple(N: int) -> Graph:
    """Generates N random 2D points and assigns binary labels based on whether the x-coordinate is less than 0.5.

    Args:
    ----
        N (int): The number of points to generate.

    Returns:
    -------
        Graph: A graph object containing the points and their corresponding binary labels (1 if x < 0.5, else 0).

    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        y1 = 1 if x_1 < 0.5 else 0
        y.append(y1)
    return Graph(N, X, y)


def diag(N: int) -> Graph:
    """Generates N random 2D points and assigns binary labels based on whether the sum of x and y coordinates is less than 0.5.

    Args:
    ----
        N (int): The number of points to generate.

    Returns:
    -------
        Graph: A graph object containing the points and their corresponding binary labels (1 if x + y < 0.5, else 0).

    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        y1 = 1 if x_1 + x_2 < 0.5 else 0
        y.append(y1)
    return Graph(N, X, y)


def split(N: int) -> Graph:
    """Generates N random 2D points and assigns binary labels based on whether the x-coordinate is outside the range [0.2, 0.8].

    Args:
    ----
        N (int): The number of points to generate.

    Returns:
    -------
        Graph: A graph object containing the points and their corresponding binary labels (1 if x < 0.2 or x > 0.8, else 0).

    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        y1 = 1 if x_1 < 0.2 or x_1 > 0.8 else 0
        y.append(y1)
    return Graph(N, X, y)


def xor(N: int) -> Graph:
    """Generates N random 2D points and assigns binary labels based on an XOR-like rule for the coordinates.

    Args:
    ----
        N (int): The number of points to generate.

    Returns:
    -------
        Graph: A graph object containing the points and their corresponding binary labels (1 if the points follow an XOR pattern, else 0).

    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        y1 = 1 if ((x_1 < 0.5 and x_2 > 0.5) or (x_1 > 0.5 and x_2 < 0.5)) else 0
        y.append(y1)
    return Graph(N, X, y)


def circle(N: int) -> Graph:
    """Generates N random 2D points and assigns binary labels based on whether the point lies outside a circle centered at (0.5, 0.5) with radius sqrt(0.1).

    Args:
    ----
        N (int): The number of points to generate.

    Returns:
    -------
        Graph: A graph object containing the points and their corresponding binary labels (1 if outside the circle, else 0).

    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        x1, x2 = (x_1 - 0.5, x_2 - 0.5)
        y1 = 1 if x1 * x1 + x2 * x2 > 0.1 else 0
        y.append(y1)
    return Graph(N, X, y)


def spiral(N: int) -> Graph:
    """Generates N points arranged in two interleaving spirals, with corresponding binary labels for each spiral.

    Args:
    ----
        N (int): The total number of points to generate (evenly split between the two spirals).

    Returns:
    -------
        Graph: A graph object containing the points arranged in two spirals and their corresponding binary labels (0 for one spiral, 1 for the other).

    """

    def x(t: float) -> float:
        return t * math.cos(t) / 20.0

    def y(t: float) -> float:
        return t * math.sin(t) / 20.0

    X = [
        (x(10.0 * (float(i) / (N // 2))) + 0.5, y(10.0 * (float(i) / (N // 2))) + 0.5)
        for i in range(5 + 0, 5 + N // 2)
    ]
    X = X + [
        (y(-10.0 * (float(i) / (N // 2))) + 0.5, x(-10.0 * (float(i) / (N // 2))) + 0.5)
        for i in range(5 + 0, 5 + N // 2)
    ]
    y2 = [0] * (N // 2) + [1] * (N // 2)
    return Graph(N, X, y2)


datasets = {
    "Simple": simple,
    "Diag": diag,
    "Split": split,
    "Xor": xor,
    "Circle": circle,
    "Spiral": spiral,
}
