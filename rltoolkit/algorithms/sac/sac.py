import copy
import logging
from itertools import chain

import numpy as np
import torch
from rltoolkit import config
from rltoolkit.algorithms.ddpg import DDPG
from rltoolkit.algorithms.sac.models import SAC_Actor, SAC_Critic
from torch.nn import functional as F

logger = logging.getLogger(__name__)


class SAC(DDPG):
    def __init__(
        self,
        alpha_lr: float = config.ALPHA_LR,
        alpha: float = config.ALPHA,
        tau: float = config.TAU,
        pi_update_freq: int = config.PI_UPDATE_FREQ,
        act_noise: float = 0,
        *args,
        **kwargs,
    ):
        f"""Soft Actor-Critic implementation

        Args:
            alpha_lr (float, optional): Learning rate of the alpha.
                Defaults to { config.ALPHA_LR }.
            alpha (float, optional): Initial alpha value. Defaults to { config.ALPHA }.
            pi_update_freq (int, optional): Frequency of policy updates
                (in SAC updates). Defaults to { config.PI_UPDATE_FREQ }.
            act_noise (float, optional): Actions noise multiplier.
                Defaults to { 0 }.
            actor_lr (float, optional): Learning rate of the actor.
                Defaults to { config.DDPG_LR }.
            critic_lr (float, optional): Learning rate of the critic.
                Defaults to { config.DDPG_LR }.
            tau (float, optional): Tau coefficient for polyak averaging.
                Defaults to { config.TAU }.
            update_batch_size (int, optional): Batch size for gradient step.
                Defaults to { config.UPDATE_BATCH_SIZE }.
            buffer_size (int, optional): Size of replay buffer.
                Defaults to { config.BUFFER_SIZE }.
            random_frames (int, optional): Number of frames with random actions at
                the beggining. Defaults to { config.RANDOM_FRAMES }.
            update_freq (int, optional): Freqency of SAC updates (in frames).
                Defaults to { config.UPDATE_FREQ }.
            grad_steps (int, optional): Number of SAC updates for one step.
                Defaults to { config.GRAD_STEPS }.
            env_name (str, optional): Name of the gym environment.
                Defaults to { config.ENV_NAME }.
            gamma (float, optional): Discount factor. Defaults to { config.GAMMA }.
            stats_freq (int, optional): Frequency of logging the progress.
                Defaults to { config.STATS_FREQ }.
            batch_size (int, optional): Number of frames used for one algorithm step
                (could be higher because batch collection stops when rollout ends).
                Defaults to { config.BATCH_SIZE }.
            iterations (int, optional): Number of algorithms iterations.
                Defaults to { config.ITERATIONS }.
            max_frames (int, optional): Limit of frames for training.
                Defaults to { None }.
            return_done (Union[int, None], optional): target return, which will stop
                training if reached. Defaults to { config.RETURN_DONE }.
            log_dir (str, optional): Path for basic logs which includes final model.
                Defaults to { config.LOG_DIR }.
            use_gpu (bool, optional): Use CUDA. Defaults to { config.USE_GPU }.
            tensorboard_dir (Union[str, None], optional): Path to tensorboard logs.
                Defaults to { config.TENSORBOARD_DIR }.
            tensorboard_comment (str, optional): Comment for tensorboard files.
                Defaults to { config.TENSORBOARD_COMMENT }.
            verbose (int, optional): Verbose level. Defaults to { config.VERBOSE }.
            render (bool, optional): Render rollouts to tensorboard.
                Defaults to { config.RENDER }.
        """
        super().__init__(*args, **kwargs)
        self._actor = None
        self.actor_optimizer = None
        self._critic_1 = None
        self.critic_1_optimizer = None
        self.critic_1_targ = None
        self._critic_2 = None
        self.critic_2_optimizer = None
        self.critic_2_targ = None

        self.alpha_lr = alpha_lr
        self.alpha = alpha
        self.pi_update_freq = pi_update_freq

        self.actor = SAC_Actor(self.ob_dim, self.ac_lim, self.ac_dim, self.discrete)
        self.critic_1 = SAC_Critic(self.ob_dim, self.ac_dim, self.discrete)
        self.critic_2 = SAC_Critic(self.ob_dim, self.ac_dim, self.discrete)

        self.loss = {"actor": 0.0, "critic_1": 0.0, "critic_2": 0.0}
        new_hparams = {
            "hparams/alpha_lr": self.alpha_lr,
            "hparams/alpha": self.alpha,
            "hparams/pi_update_freq": self.pi_update_freq,
        }
        self.hparams.update(new_hparams)

        self.target_entropy = -torch.prod(
            torch.tensor(self.ac_dim, dtype=torch.float32)
        ).item()
        self.log_alpha = torch.tensor(
            np.log(self.alpha), requires_grad=True, device=self.device
        )
        self.alpha_opt = self.opt([self.log_alpha], lr=alpha_lr)

    @property
    def actor(self):
        return self._actor

    @actor.setter
    def actor(self, model: torch.nn.Module):
        self._actor, self.actor_optimizer = self.set_model(model, self.actor_lr)

    @property
    def critic_1(self):
        return self._critic_1

    @critic_1.setter
    def critic_1(self, model: torch.nn.Module):
        self._critic_1, self.critic_1_optimizer = self.set_model(model, self.critic_lr)
        self.critic_1_targ = copy.deepcopy(self._critic_1)

    @property
    def critic_2(self):
        return self._critic_2

    @critic_2.setter
    def critic_2(self, model: torch.nn.Module):
        self._critic_2, self.critic_2_optimizer = self.set_model(model, self.critic_lr)
        self.critic_2_targ = copy.deepcopy(self._critic_2)

    def compute_qfunc_targ(
        self, reward: torch.Tensor, next_obs: torch.Tensor, done: torch.Tensor
    ):
        """Compute targets for Q-functions

        Args:
            reward (torch.Tensor): batch of rewards
            next_obs (torch.Tensor): batch of next observations
            done (torch.Tensor): batch of done

        Returns:
            torch.Tensor: Q-function targets for the batch
        """
        pass

    def compute_pi_loss(
        self,
        obs: torch.Tensor,
        sampled_action: torch.Tensor,
        sampled_logprob: torch.Tensor,
    ):
        """Loss for the policy

        Args:
            obs (torch.Tensor): batch of observations
            sampled_action (torch.Tensor): actions sampled from policy
            sampled_logprob (torch.Tensor): log-probabilities of actions

        Returns:
            torch.Tensor: policy loss
        """
        pass

    def update_target_q(self):
        """Update target networks with Polyak averaging"""
        pass

    def compute_alpha_loss(self, sampled_logprob: torch.Tensor):
        """Compute loss for temperature update

        Args:
            sampled_logprob (torch.Tensor): batch of sampled log-probabilities
                from the actor

        Returns:
            torch.Tensor: loss for temperature (alpha)
        """
        pass

    def update(
        self,
        obs: torch.Tensor,
        next_obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
    ):
        """Soft Actor-Critic update:

        Args:
            obs (torch.Tensor): observations tensor
            next_obs (torch.Tensor): next observations tensor
            action (torch.Tensor): actions tensor
            reward (torch.Tensor): rewards tensor
            done (torch.Tensor): dones tensor
        """
        pass

    def add_tensorboard_logs(self, *args, **kwargs):
        super().add_tensorboard_logs(*args, **kwargs)
        if self.debug_mode:
            self.tensorboard_writer.log_sac_alpha(self.iteration, self.alpha)

    def collect_params_dict(self):
        params_dict = {}
        params_dict["actor"] = self.actor.state_dict()
        params_dict["critic_1"] = self.critic_1.state_dict()
        params_dict["critic_2"] = self.critic_2.state_dict()
        params_dict["obs_mean"] = self.replay_buffer.obs_mean
        params_dict["obs_std"] = self.replay_buffer.obs_std
        return params_dict

    def apply_params_dict(self, params_dict):
        self.actor.load_state_dict(params_dict["actor"])
        self.critic_1.load_state_dict(params_dict["critic_1"])
        self.critic_2.load_state_dict(params_dict["critic_2"])
        self.replay_buffer.obs_mean = params_dict["obs_mean"]
        self.replay_buffer.obs_std = params_dict["obs_std"]

    def save_model(self, save_path=None) -> str:
        if self.filename is None and save_path is None:
            raise AttributeError
        elif save_path is None:
            save_path = str(self.log_path)

        torch.save(self._actor.state_dict(), save_path + "_actor_model.pt")
        torch.save(self._critic_1.state_dict(), save_path + "_critic_1_model.pt")
        torch.save(self._critic_2.state_dict(), save_path + "_critic_2_model.pt")
        return save_path


if __name__ == "__main__":
    with torch.cuda.device(1):
        model = SAC(
            env_name="HalfCheetah-v2",
            iterations=200,
            gamma=0.99,
            batch_size=1000,
            stats_freq=5,
            test_episodes=2,
            update_batch_size=100,
            update_freq=50,
            grad_steps=50,
            # random_frames=10000,
            use_gpu=True,
            obs_norm=False,
            tensorboard_dir="logs_norm",
            tensorboard_comment="",
        )
        model.train()
