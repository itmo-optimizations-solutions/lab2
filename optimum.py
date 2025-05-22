import optuna
from main import *

X0 = np.array([-1.0, 5.0])
FUNC = NaryFunc(rosenbrock)
descent = bfgs_descent

def objective(trial: optuna.Trial) -> tuple[float, int, int]:
    # a = trial.suggest_uniform("a", 1e-6, 1.0)
    # q = trial.suggest_uniform("q", 1e-6, 0.9)
    # c = trial.suggest_loguniform("c", 1e-6, 0.5)
    # learning = armijo_rule_gen(a, q, c)

    # a = trial.suggest_uniform("a", 0, 1)
    # b = trial.suggest_uniform("b", 0, 1)
    # learning = dichotomy_gen(a, b)

    # a = trial.suggest_uniform("a", 1e-6, 1.0)
    # c1 = trial.suggest_uniform("c1", 1e-6, 1.0)
    # c2 = trial.suggest_loguniform("c2", 1e-6, 1.0)
    # learning = wolfe_rule_gen(a, c1, c2)

    a = trial.suggest_uniform("a", 0.0, 10.0)
    learning = constant(a)

    res, grad_count, steps, _, _ = descent(FUNC, X0, learning)
    return FUNC(res), grad_count, steps

if __name__ == "__main__":
    study = optuna.create_study(directions=["minimize", "minimize", "minimize"])
    study.optimize(objective, n_trials=100, timeout=600)

    for t in study.best_trials:
        print(f"Trial #{t.number}")
        print("  values =", t.values)
        print("  params =", t.params)
        # print("  Result = ", descent(FUNC, X0, armijo_rule_gen(
        #     t.params["a"],
        #     t.params["q"],
        #     t.params["c"]))[:4])
        # print("  Result = ", descent(FUNC, X0, dichotomy_gen(
        #     t.params["a"],
        #     t.params["b"]))[:4])
        # print("  Result = ", descent(FUNC, X0, wolfe_rule_gen(
        #     t.params["a"],
        #     t.params["c1"],
        #     t.params["c2"]))[:4])
        print("  Result = ", descent(FUNC, X0, constant(
            t.params["a"]))[:4])
        print()
