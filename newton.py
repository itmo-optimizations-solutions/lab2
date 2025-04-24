from nary import *
from scipy.optimize import minimize
from typing import Tuple

def newton_cg(func: NaryFunc, start: Vector) -> Tuple[Vector, int, int]:
    res = minimize(
        fun=func,
        x0=start,
        method='Newton-CG',
        jac=lambda x: func.gradient(x),
        hess=lambda x: func.hessian(x),
        options={'maxiter': 1000, 'xtol': 1e-6},
    )
    return res.x, func.g_count, res.nit

def bfgs(func: NaryFunc, start: Vector) -> Tuple[Vector, int, int]:
    res = minimize(
        fun=func,
        x0=start,
        method='BFGS',
        jac=lambda x: func.gradient(x),
        options={'maxiter': 1000, 'gtol': 1e-6},
    )
    return res.x, func.g_count, res.nit
