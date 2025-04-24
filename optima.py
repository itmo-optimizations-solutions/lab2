import numpy as np
import optuna
from numpy import ndarray
from optuna.trial import FrozenTrial

from plotly.io import show
from plot import *
from main import (
    newton_descent,
    constant,
    exponential_decay,
    polynomial_decay,
    armijo_rule_gen,
    wolfe_rule_gen,
    scipy_armijo,
    scipy_wolfe,
    dichotomy_gen,
    NaryFunc,
    rosenbrock,
)

X0 = np.array([0.0, 5.0])
FUNC = NaryFunc(rosenbrock)

def objective(trial: optuna.Trial) -> tuple[float, int, int]:

    α0 = trial.suggest_uniform("armijo_alpha", 1e-6, 1.0)
    q  = trial.suggest_uniform("armijo_q",    1e-6, 0.9)
    c  = trial.suggest_loguniform("armijo_c", 1e-6, 0.5)
    learning = armijo_rule_gen(α0, q, c)

    res, grad_count, steps, _ = newton_descent(
        FUNC, X0, learning
    )

    return FUNC(res), grad_count, steps

if __name__ == "__main__":
    study = optuna.create_study(directions=["minimize", "minimize", "minimize"])
    study.optimize(objective, n_trials=100, timeout=600)

    for t in study.best_trials:
        print(f"Trial #{t.number}")
        print("  values =", t.values)
        print("  params =", t.params)
        print()

    fig = optuna.visualization.plot_pareto_front(
        study,
        target_names=["grad_count", "steps", "f_value"]
    )
    show(fig)
    # # По желанию можно сохранить результаты:
    #
    # print("\nRe-running newton_descent with best params...")
    #
    # alpha0 = best_trial.params["armijo_alpha"]
    # q = best_trial.params["armijo_q"]
    # c = best_trial.params["armijo_c"]
    # best_learning = armijo_rule_gen(α=0.999944970635587, q=0.08285586209188826, c=0.0007192571029004037)
    #
    # x_opt, grad_count, steps, trajectory = newton_descent(
    #     FUNC, X0, best_learning
    # )
    #
    # print(f"Optimal x: {x_opt}")
    # print(f"Gradient calls: {grad_count}, Steps: {steps}")
    #
    # plot_gradient(FUNC, len(X0) == 1, len(X0) == 2, trajectory)
