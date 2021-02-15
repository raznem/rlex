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
    ###############################################################
    # This time your task is to implement essential parts of the
    # SAC. Same as before, this SAC class is almost ready, the
    # only part left for you is to finish the update implementation
    # Also note that this time our actor is stochastic, so forward
    # is much more complicated than in case of DDPG. You can check
    # the details of implementatation in the models file:
    #   rltoolkit/algorithms/sac/models.py
    ###############################################################
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
        ###############################################################
        # Implement function for calculating target or your Critic from
        # the (r, d, s') tuple. See line 12 in the:
        #   https://spinningup.openai.com/en/latest/algorithms/sac.html
        #
        # Hint: This step is very similar to the Critic target from DDPG,
        # so make sure you have correct gradient flow.
        # Here and late refer to:
        #   self.actor_targ: target actor
        #   self._critic_1: the first critic
        #   self._critic_2: the second critic
        #   self._critic_1_targ: target network of the first critic
        #   self._critic_2_targ: target network of the second critic
        #   self._actor: actor which return tuple (action, action logprob)
        #   self.gamma: gamma discount factor
        #   self.alpha: entropy temperature
        ###############################################################
        # Your code is here #
        qfunc_target =

        # End of your code #
        return qfunc_target

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
        ###############################################################
        # Compute policy loss. See line 14 in the pseudocode from:
        #   https://spinningup.openai.com/en/latest/algorithms/sac.html
        ###############################################################
        # Your code is here #
        loss = 
        # End of your code #
        return loss

    def update_target_q(self):
        """Update target networks with Polyak averaging"""
        ###############################################################
        # Same Polyak averaging as in DDPG
        ###############################################################
        rho = 1 - self.tau
        with torch.no_grad():
            # Polyak averaging:
            critics_params = chain(
                self._critic_1.parameters(), self._critic_2.parameters()
            )
            targets_params = chain(
                self.critic_1_targ.parameters(), self.critic_2_targ.parameters()
            )
            for q_params, targ_params in zip(critics_params, targets_params):
                targ_params.data.mul_(rho)
                targ_params.data.add_((1 - rho) * q_params.data)

    def compute_alpha_loss(self, sampled_logprob: torch.Tensor):
        """Compute loss for temperature update

        Args:
            sampled_logprob (torch.Tensor): batch of sampled log-probabilities
                from the actor

        Returns:
            torch.Tensor: loss for temperature (alpha)
        """
        ###############################################################
        # Loss for the learnable alpha from the second version of SAC
        # If you are interested in details see part 5 in the second SAC
        # paper: https://arxiv.org/abs/1812.05905
        ###############################################################
        sampled_logprob = sampled_logprob.detach()
        alpha_loss = self.log_alpha.exp() * (-sampled_logprob - self.target_entropy)
        return alpha_loss.mean()

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
        ###############################################################
        # After implementing particalar parts of the SAC update we
        # will join them similarly as we joined in DDPG task.
        # At the beggining implement q-function update.
        # Refer to the line 13 from OpenAI pseudocode:
        #   https://spinningup.openai.com/en/latest/algorithms/sac.html
        # Hint: comparing to the DDPG you have additional critic
        ###############################################################
        # Your code is here #
        loss_q1 = 
        loss_q2 = 
        # End of your code #

        self.loss["critic_1"] = loss_q1.item()
        self.loss["critic_2"] = loss_q2.item()

        self.critic_1_optimizer.zero_grad()
        loss_q1.backward()
        self.critic_1_optimizer.step()

        self.critic_2_optimizer.zero_grad()
        loss_q2.backward()
        self.critic_2_optimizer.step()

        ###############################################################
        # Update of the policy network.
        # You need to finish line 14 from the SAC pseudocode:
        #   https://spinningup.openai.com/en/latest/algorithms/sac.html
        # Hint: your actor is stochastic
        ###############################################################
        self._critic_1.eval()
        self._critic_2.eval()
        # Your code is here #
        loss = 
        # End of your code #
        self.loss["actor"] = loss.item()

        self.actor_optimizer.zero_grad()
        loss.backward()
        self.actor_optimizer.step()

        ###############################################################
        # Update of the target networks
        ###############################################################
        self.update_target_q()

        self._critic_1.train()
        self._critic_2.train()

        ###############################################################
        # Update of the entropy temperature
        ###############################################################
        assert 'sampled_logprob' in locals(), "define sampled_logprob"
        alpha_loss = self.compute_alpha_loss(sampled_logprob)

        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()
        self.alpha = self.log_alpha.exp().item()

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
    model = SAC(
        env_name="Pendulum-v0",
        iterations=100,
        gamma=0.99,
        batch_size=50,
        stats_freq=5,
        test_episodes=2,
    )
    model.train()
