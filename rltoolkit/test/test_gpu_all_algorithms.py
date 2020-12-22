import torch

from rltoolkit import A2C, DDPG, PPO, SAC

ITERATIONS = 5
USE_GPU = True
STATS_FREQ = 1


def test_a2c_discrete_action():
    if not torch.cuda.is_available():
        return

    model = A2C(
        use_gpu=USE_GPU,
        iterations=ITERATIONS,
        stats_freq=STATS_FREQ,
        tensorboard_comment="test",
    )
    model.train()


def test_a2c_continuous_action():
    if not torch.cuda.is_available():
        return

    model = A2C(
        env_name="Pendulum-v0",
        use_gpu=USE_GPU,
        iterations=ITERATIONS,
        stats_freq=STATS_FREQ,
        tensorboard_comment="test",
    )
    model.train()


def test_ddpg_continuous_action():
    if not torch.cuda.is_available():
        return

    model = DDPG(
        env_name="Pendulum-v0",
        use_gpu=USE_GPU,
        iterations=ITERATIONS,
        stats_freq=STATS_FREQ,
        tensorboard_comment="test",
    )
    model.train()


def test_ppo_discrete_action():
    if not torch.cuda.is_available():
        return

    model = PPO(
        use_gpu=USE_GPU,
        iterations=ITERATIONS,
        stats_freq=STATS_FREQ,
        tensorboard_comment="test",
    )
    model.train()


def test_ppo_continuous_action():
    if not torch.cuda.is_available():
        return

    model = PPO(
        env_name="Pendulum-v0",
        use_gpu=USE_GPU,
        iterations=ITERATIONS,
        stats_freq=STATS_FREQ,
        tensorboard_comment="test",
    )
    model.train()


def test_sac_continuous_action():
    if not torch.cuda.is_available():
        return

    model = SAC(
        env_name="Pendulum-v0",
        use_gpu=USE_GPU,
        iterations=ITERATIONS,
        stats_freq=STATS_FREQ,
        tensorboard_comment="test",
    )
    model.train()
