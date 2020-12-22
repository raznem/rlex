# RL config
ENV_NAME = "CartPole-v0"
ITERATIONS = 200
GAMMA = 0.95
BATCH_SIZE = 200
STATS_FREQ = 20
TEST_EPISODES = None
RETURN_DONE = None
LOG_DIR = None
USE_GPU = False
TENSORBOARD_DIR = None
TENSORBOARD_COMMENT = ""
VERBOSE = 1
RENDER = False
DEBUG_MODE = True

# A2C config
A_LR = 3e-3
C_LR = 3e-4
NUM_TARGET_UPDATES = 10
NUM_CRITIC_UPDATES = 10
NORMALIZE_ADV = True

# PPO config
PPO_EPSILON = 0.2
GAE_LAMBDA = 0.95
PPO_MAX_KL_DIV = 0.15
PPO_MAX_EPOCHS = 50
PPO_BATCH_SIZE = 1000
PPO_ENTROPY = 0.00

# DDPG
DDPG_LR = 1e-3
TAU = 0.005
UPDATE_BATCH_SIZE = 100
BUFFER_SIZE = int(1e6)
RANDOM_FRAMES = 100
UPDATE_FREQ = 50
GRAD_STEPS = 50
ACT_NOISE = 0.1

# SAC config
ALPHA_LR = 1e-3
ALPHA = 0.2
PI_UPDATE_FREQ = 1

# Norm config
MAX_ABS_OBS_VALUE = 10
NORM_ALPHA = 0.99
OBS_NORM = False

SHORTNAMES = {
    "hparams/type": "",
    "hparams/gamma": "g",
    "hparams/batch_size": "bs",
    "hparams/actor_lr": "a_lr",
    "hparams/critic_lr": "c_lr",
    "hparams/critic_num_target_updates": "c_target_u",
    "hparams/num_critic_updates_per_target": "c_updates_pt",
    "hparams/normalize_adv": "norm",
    "hparams/buffer_batches": "bb",
    "hparams/ppo_epsilon": "ppo_eps",
    "hparams/gae_lambda": "gae_l",
    "hparams/kl_div_threshold": "kl_thr",
    "hparams/max_ppo_epochs": "ppo_e",
    "hparams/ppo_batch_size": "ppo_bs",
    "hparams/alpha": "al",
    "hparams/tau": "tau",
    "hparams/update_batch_size": "ubs",
    "hparams/buffer_size": "bu_s",
    "hparams/update_after": "ua",
    "hparams/random_frames": "rf",
    "hparams/update_freq": "ufreq",
    "hparams/pi_update_freq": "pi_ufreq",
    "hparams/grad_steps": "gs",
    "hparams/act_noise": "noise",
}
