from rltoolkit import DDPG, PPO, SAC, EvalsWrapper

ITERATIONS = 3
MAX_FRAMES = 2e2
STATS_FREQ = 1
EVALS = 2
ENV = "Pendulum-v0"
BATCH_SIZE = 100
OFF_POLICY_BUFFER_SIZE = 100


def test_ppo_evals():
    evals_wrapper = EvalsWrapper(
        Algo=PPO,
        evals=EVALS,
        tensorboard_dir="test_logs",
        max_frames=MAX_FRAMES,
        iterations=ITERATIONS,
        stats_freq=STATS_FREQ,
        batch_size=BATCH_SIZE,
        ppo_batch_size=32,
        max_ppo_epochs=2,
        env_name=ENV,
    )
    evals_wrapper.perform_evaluations()
    evals_wrapper.update_tensorboard()


def test_ddpg_evals():
    evals_wrapper = EvalsWrapper(
        Algo=DDPG,
        evals=EVALS,
        tensorboard_dir="test_logs",
        max_frames=MAX_FRAMES,
        iterations=ITERATIONS,
        stats_freq=STATS_FREQ,
        batch_size=BATCH_SIZE,
        env_name=ENV,
        buffer_size=OFF_POLICY_BUFFER_SIZE,
    )
    evals_wrapper.perform_evaluations()
    evals_wrapper.update_tensorboard()


def test_sac_evals():
    evals_wrapper = EvalsWrapper(
        Algo=SAC,
        evals=EVALS,
        tensorboard_dir="test_logs",
        max_frames=MAX_FRAMES,
        iterations=ITERATIONS,
        stats_freq=STATS_FREQ,
        batch_size=BATCH_SIZE,
        env_name=ENV,
        buffer_size=OFF_POLICY_BUFFER_SIZE,
    )
    evals_wrapper.perform_evaluations()
    evals_wrapper.update_tensorboard()
