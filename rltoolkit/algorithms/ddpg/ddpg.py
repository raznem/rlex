import copy
import logging
from itertools import chain

import numpy as np
import torch
from rltoolkit import config
from rltoolkit.algorithms.ddpg.models import Actor, Critic
from rltoolkit.buffer import ReplayBuffer
from rltoolkit.rl import RL
from rltoolkit.utils import measure_time
from torch.nn import functional as F

logger = logging.getLogger(__name__)


class DDPG(RL):
    ###############################################################
    # This time your task is to implement essential parts of the
    # DDPG algorithm. You have this DDPG class almost ready, the
    # only part left for you to finish the update implementation
    # and generation of noisy actions.
    ###############################################################
    def __init__(
        self,
        actor_lr: float = config.DDPG_LR,
        critic_lr: float = config.DDPG_LR,
        tau: float = config.TAU,
        update_batch_size: int = config.UPDATE_BATCH_SIZE,
        buffer_size: int = config.BUFFER_SIZE,
        random_frames: int = config.RANDOM_FRAMES,
        update_freq: int = config.UPDATE_FREQ,
        grad_steps: int = config.GRAD_STEPS,
        act_noise: float = config.ACT_NOISE,
        obs_norm: bool = config.OBS_NORM,
        *args,
        **kwargs,
    ):
        f"""Deep Deterministic Policy Gradient implementation

        Args:
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
            act_noise (float, optional): Actions noise multiplier.
                Defaults to { config.ACT_NOISE }.
            obs_norm (bool, optional): Observation normalization.
                Defaults to { False }.
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
        assert not self.discrete, "DDPG works only on continuous actions space"
        self._actor = None
        self.actor_optimizer = None
        self._actor_targ = None
        self._critic = None
        self.critic_optimizer = None
        self.critic_targ = None

        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.tau = tau
        self.update_batch_size = update_batch_size
        self.buffer_size = buffer_size
        self.random_frames = random_frames
        self.update_freq = update_freq
        self.grad_steps = grad_steps
        self.act_noise = act_noise
        self.obs_norm = obs_norm
        self.obs_mean, self.obs_std = self._get_initial_obs_mean_std(self.obs_norm)

        self.actor = Actor(self.ob_dim, self.ac_lim, self.ac_dim)
        self.critic = Critic(self.ob_dim, self.ac_dim)

        self.replay_buffer = ReplayBuffer(
            self.buffer_size,
            self.ob_dim,
            self.ac_dim,
            discrete=self.discrete,
            dtype=torch.float32,
            device=self.device,
            obs_norm=self.obs_norm,
        )

        self.loss = {"actor": 0.0, "critic": 0.0}
        new_hparams = {
            "hparams/actor_lr": self.actor_lr,
            "hparams/critic_lr": self.critic_lr,
            "hparams/tau": self.tau,
            "hparams/update_batch_size": self.update_batch_size,
            "hparams/buffer_size": self.buffer_size,
            "hparams/random_frames": self.random_frames,
            "hparams/update_freq": self.update_freq,
            "hparams/grad_steps": self.grad_steps,
            "hparams/act_noise": self.act_noise,
            "hparams/obs_norm": self.obs_norm,
        }
        self.hparams.update(new_hparams)

    def set_model(self, model, lr):
        model.to(device=self.device)
        optimizer = self.opt(model.parameters(), lr=lr)
        return model, optimizer

    @property
    def actor(self):
        return self._actor

    @actor.setter
    def actor(self, model: torch.nn.Module):
        self._actor, self.actor_optimizer = self.set_model(model, self.actor_lr)
        self.actor_targ = copy.deepcopy(self._actor)
        for p in self.actor_targ.parameters():
            p.requires_grad = False

    @property
    def critic(self):
        return self._critic

    @critic.setter
    def critic(self, model: torch.nn.Module):
        self._critic, self.critic_optimizer = self.set_model(model, self.critic_lr)
        self.critic_targ = copy.deepcopy(self._critic)
        for p in self.critic_targ.parameters():
            p.requires_grad = False

    @measure_time
    def perform_iteration(self):
        f"""Single train step of algorithm

        Returns:
            Memory: Buffer filled with one batch
            float: Time taken for evaluation
        """
        self.collect_batch_and_train(self.batch_size)
        self.replay_buffer = self.update_obs_mean_std(self.replay_buffer)
        return self.replay_buffer.last_rollout()

    def noise_act(self, obs: torch.Tensor, act_noise: float) -> torch.Tensor:
        """Noisy actor wrapper

        Args:
            obs (torch.Tensor): observation tansor
            act_noise (float): noise multiplier

        Returns:
            torch.Tensor: noisy actions
        """
        ###############################################################
        # At the beginning implement function that generates noisy
        # actions from observations. For details, see line 4 in the
        # pseudocode from https://spinningup.openai.com/en/latest/algorithms/ddpg.html
        # Hint: use np.clip for action clipping.
        # Also, note that actor returns 2 values, first is an action and
        #   the second is a placeholder.
        ###############################################################
        action, _ = self._actor.act(obs)
        action = action.cpu()
        action_size = self.ac_dim
        action_high = self.ac_lim.cpu()
        action_low = -self.ac_lim.cpu()
        # Your code is here #
        action = 
        # End of your code #
        return action

    def initial_act(self, obs) -> torch.Tensor:
        f"""Randomly generated actions for the warmup stage of the training

        Args:
            obs (torch.Tensor): observation tensor

        Returns:
            torch.Tensor: action tensor
        """
        action = torch.tensor(self.env.action_space.sample()).unsqueeze(0)
        return action

    def collect_batch_and_train(self, batch_size: int, *args, **kwargs):
        f"""Perform full rollouts and collect samples till batch_size number of steps
            will be added to the replay buffer

        Args:
            batch_size (int): number of samples to collect and train
            *args, **kwargs: arguments for make_update
        """
        ###############################################################
        # Off-policy algorithms are usually updated more often than
        # on-policy. Hence you have slightly different implementation
        # of data collection. This one doesn't wait for the end of the
        # episode to make update, but can do this update even during
        # the rollout. Note that this approach is not limited to off-
        # policy algortihms and could be used for A2C and PPO as well.
        # You will work only on the last line of this function -
        # DDPG update.
        ###############################################################
        collected = 0
        while collected < batch_size:
            self.stats_logger.rollouts += 1

            obs = self.env.reset()
            # end - end of the episode from the perspective of the simulation
            # done - end of the episode from the perspective of the model
            end = False
            obs = self.process_obs(obs)
            prev_idx = self.replay_buffer.add_obs(obs)
            ep_len = 0

            while not end:
                obs = self.replay_buffer.normalize(obs)
                if self.stats_logger.frames < self.random_frames:
                    action = self.initial_act(obs)
                else:
                    action = self.noise_act(obs, self.act_noise)
                action_proc = self.process_action(action, obs)
                obs, rew, done, _ = self.env.step(action_proc)
                ep_len += 1
                end = done
                done = False if ep_len == self.max_ep_len else done

                obs = self.process_obs(obs)
                next_idx = self.replay_buffer.add_obs(obs)
                self.replay_buffer.add_timestep(
                    prev_idx, next_idx, action, rew, done, end
                )
                prev_idx = next_idx
                self.stats_logger.frames += 1
                collected += 1

                self.make_update(*args, **kwargs)

    def update_condition(self):
        """Method that checks whether update should be performed.

        Returns:
            bool: indication of update time
        """
        return (
            len(self.replay_buffer) > self.update_batch_size
            and self.stats_logger.frames % self.update_freq == 0
        )

    def make_update(self):
        """Perform update self.grad_steps times."""
        if self.update_condition():
            for _ in range(self.grad_steps):
                batch = self.replay_buffer.sample_batch(
                    self.update_batch_size, self.device
                )
                self.update(*batch)

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
        #   https://spinningup.openai.com/en/latest/algorithms/ddpg.html
        #
        # Hint: This step is very similar to the Critic target from A2C,
        # so make sure you have correct gradient flow.
        # Here and late refer to:
        #   self.actor_targ: target actor
        #   self.critic_targ: target critic
        #   self._actor: actor
        #   self._critic: critic
        #   self.gamma: gamma discount factor
        # P. S. This time critic estimates Q-function, thus it has 2
        #   arguments.
        ###############################################################
        # Your code is here #
        qfunc_target = 
        # End of your code #
        return qfunc_target

    def compute_pi_loss(self, obs):
        """Loss for the policy

        Args:
            obs (torch.Tensor): batch of observations

        Returns:
            torch.Tensor: policy loss
        """
        ###############################################################
        # Compute policy loss. See line 14 in the pseudocode from:
        #   https://spinningup.openai.com/en/latest/algorithms/ddpg.html
        # Hint: This step is very similar to the Critic target from A2C,
        # so make sure you have correct gradient flow.
        ###############################################################
        # Your code is here #
        loss = 
        # End of your code #
        return loss

    def update_target_nets(self):
        """Update target networks with Polyak averaging"""
        ###############################################################
        # See line 15 in the pseudocode from:
        #   https://spinningup.openai.com/en/latest/algorithms/ddpg.html
        #
        # To overwrite parameters in the pytorch network use attribute
        #   <parameters_name>.data
        ###############################################################
        rho = 1 - self.tau
        with torch.no_grad():
            # Polyak averaging:
            learned_params = chain(self._critic.parameters(), self._actor.parameters())
            targets_params = chain(
                self.critic_targ.parameters(), self.actor_targ.parameters()
            )
            for params, targ_params in zip(learned_params, targets_params):
                # Your code is here #
                pass
                # End of your code #

    def update(
        self,
        obs: torch.Tensor,
        next_obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
    ):
        """DDPG update step

        Args:
            obs (torch.Tensor): observations tensor
            next_obs (torch.Tensor): next observations tensor
            action (torch.Tensor): actions tensor
            reward (torch.Tensor): rewards tensor
            done (torch.Tensor): dones tensor
        """
        ###############################################################
        # After implementing particalar parts of the DDPG update we
        # will join them in the update function.
        # At the beggining implement q-function update.
        # Refer to the line 13 from OpenAI pseudocode:
        #   https://spinningup.openai.com/en/latest/algorithms/ddpg.html
        ###############################################################
        # Your code is here #
        loss_q = 
        # End of your code #

        self.loss["critic"] = loss_q.item()
        self.critic_optimizer.zero_grad()
        loss_q.backward()
        self.critic_optimizer.step()

        ###############################################################
        # Update of the policy network
        ###############################################################
        self._critic.eval()
        loss = self.compute_pi_loss(obs)
        self.loss["actor"] = loss.item()
        self.actor_optimizer.zero_grad()
        loss.backward()
        self.actor_optimizer.step()

        ###############################################################
        # Update of the target networks
        ###############################################################
        self.update_target_nets()
        self._critic.train()

    def collect_params_dict(self):
        params_dict = {}
        params_dict["actor"] = self.actor.state_dict()
        params_dict["critic"] = self.critic.state_dict()
        params_dict["obs_mean"] = self.replay_buffer.obs_mean
        params_dict["obs_std"] = self.replay_buffer.obs_std
        return params_dict

    def apply_params_dict(self, params_dict):
        self.actor.load_state_dict(params_dict["actor"])
        self.critic.load_state_dict(params_dict["critic"])
        self.replay_buffer.obs_mean = params_dict["obs_mean"]
        self.replay_buffer.obs_std = params_dict["obs_std"]

    def save_model(self, save_path=None):
        if self.filename is None and save_path is None:
            raise AttributeError
        elif save_path is None:
            save_path = str(self.log_path)

        torch.save(self._actor.state_dict(), save_path + "_actor_model.pt")
        torch.save(self._critic.state_dict(), save_path + "_critic_model.pt")
        return save_path

    def process_obs(self, obs):
        f"""Pre-processing of observation before it will go to the policy

        Args:
            obs (iter): original observation from env

        Returns:
            torch.Tensor: processed observation
        """
        obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
        obs = torch.unsqueeze(obs, dim=0)
        return obs

    def process_action(self, action: torch.Tensor, obs: torch.tensor, *args, **kwargs):
        f"""Pre-processing of action before it will go the env.
        It will not be saved to the buffer.

        Args:
            action (torch.Tensor): action from the policy
            obs (torch.tensor): observations for this actions

        Returns:
            np.array: processed action
        """
        action = action.cpu().numpy()[0]
        return action

    def test(self, episodes=None):
        f"""Run deterministic policy and log average return

        Args:
            episodes (int, optional): Number of episodes for test. Defaults to { 10 }.

        Returns:
            float: mean episode reward
        """
        if episodes is None:
            episodes = self.test_episodes
        returns = []
        for j in range(episodes):
            obs = self.env.reset()
            done = False
            ep_ret = 0
            while not done:
                obs = self.process_obs(obs)
                obs = self.replay_buffer.normalize(obs)
                action, _ = self._actor.act(obs, deterministic=True)
                action_proc = self.process_action(action, obs)
                obs, r, done, _ = self.env.step(action_proc)
                ep_ret += r
            returns.append(ep_ret)

        return np.mean(returns)


if __name__ == "__main__":
    model = DDPG(
        env_name="Pendulum-v0",
        iterations=100,
        gamma=0.99,
        batch_size=50,
        stats_freq=5,
        test_episodes=2,
    )
    model.train()
