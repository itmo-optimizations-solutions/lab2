import numpy as np
from typing import Callable

Vector = np.ndarray
SYSTEM_EPS = np.sqrt(np.finfo(float).eps)


class NaryFunc:
    Type = Callable[[list[Vector]], float]
    func: Type
    g_count: int
    h_count: int

    def __init__(self, func: Type) -> None:
        self.func = func
        self.g_count = 0

    def __call__(self, x: Vector) -> float:
        return self.func(*x)

    def partial_derivative(
        self, i: int, size: int, ε: float = SYSTEM_EPS
    ) -> "NaryFunc":
        dx = np.zeros(size)
        h = max(ε, SYSTEM_EPS)
        dx[i] = h
        return NaryFunc(lambda x: (self(x + dx) - self(x - dx)) / (2 * h))

    def gradient(self, x: Vector, ε: float = SYSTEM_EPS) -> Vector:
        self.g_count += 1
        gradient = np.zeros_like(x)
        size = len(x)
        for i in range(size):
            gradient[i] = self.partial_derivative(i, size, ε)(x)
        return gradient

    def hessian(self, x: Vector, ε: float = SYSTEM_EPS) -> Vector:
        self.h_count += 1
        size = len(x)
        hessian = np.zeros((size, size))

        for i in range(size):
            df_i = self.partial_derivative(i, size, ε)
            for j in range(size):
                hessian[i, j] = df_i.partial_derivative(j, size, ε)(x)

        return hessian


Scheduling = Callable[[int], float]
Rule = Callable[[NaryFunc, Vector, Vector], float]

Learning = Scheduling | Rule
