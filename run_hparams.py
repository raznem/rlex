from itertools import product

from rltoolkit import A2C, EvalsWrapper
import multiprocessing as mp

ALGO = A2C
EVALS = 1
ITERATIONS = 200
RETURN_DONE = 190
GAMMA = [0.99]
A_LR = [1e-3, 3e-3, 1e-2, 3e-2]
C_LR = [1e-3, 3e-3, 1e-2, 3e-2]
BATCH_SIZE = [50, 200, 500]
TENSORBOARD_DIR = "tensorboard"
LOG_DIR = "basic_logs"
N_CORES = 4


def run_combination(*args, **kwargs):
    evals = EvalsWrapper(*args, **kwargs)
    evals.perform_evaluations()
    evals.update_tensorboard()


def apply_kwargs(fn, kwargs):
    return fn(**kwargs)


if __name__ == "__main__":
    combinations = product(A_LR, C_LR, BATCH_SIZE, GAMMA)
    kwargs_list = []
    for a_lr, c_lr, batch_size, gamma in combinations:
        kwargs = {
            "Algo": ALGO,
            "evals": EVALS,
            "iterations": ITERATIONS,
            "gamma": gamma,
            "actor_lr": a_lr,
            "critic_lr": c_lr,
            "batch_size": batch_size,
            "tensorboard_dir": TENSORBOARD_DIR,
            "return_done": RETURN_DONE,
            "log_dir": LOG_DIR,
            "verbose": 0,
            "render": False,
            "test_episodes": 5,
        }
        kwargs_list.append((run_combination, kwargs))

    with mp.Pool(N_CORES) as p:
        p.starmap(apply_kwargs, kwargs_list)
