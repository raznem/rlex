import logging

import torch
from rltoolkit import config, utils
from rltoolkit.algorithms.a2c.a2c import A2C
from rltoolkit.algorithms.ppo.advantage_dataset import AdvantageDataset
from rltoolkit.buffer import Memory
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


class PPO(A2C):
    ####################################################################
    # Your task is to implement PPO with Generalized Advantage Estimation
    # (GAE, https://arxiv.org/abs/1506.02438).
    # Implememnt missing methods from top to bottom.
    # Warning: you need working A2C class to finish this task.
    # You can if it's correctly implemented running `pytest rltoolkit/algorithms/a2c/`
    # in the project root directory. The same works for PPO.
    ####################################################################

    def __init__(
        self,
        epsilon: float = config.PPO_EPSILON,
        gae_lambda: float = config.GAE_LAMBDA,
        kl_div_threshold: float = config.PPO_MAX_KL_DIV,
        max_ppo_epochs: int = config.PPO_MAX_EPOCHS,
        ppo_batch_size: int = config.PPO_BATCH_SIZE,
        entropy_coef: float = config.PPO_ENTROPY,
        *args,
        **kwargs,
    ):
        f"""
        Proximal Policy Optimization implementation.

        Args:
            epsilon (float, optional): Clipping PPO parameter. Ratio between a new
                action probability and old one will be clipped to the range
                <1 - epsilon, 1 + epsilon>  Defaults to { config.PPO_EPSILON }.
            gae_lambda (float, optional): Generalized Advantage Estimation lambda
                parameter. Defaults to { config.GAE_LAMBDA }.
            kl_div_threshold (float, optional): Maximal KL divergence between old and
                new policy. If this treshold is acquired agent's update is stopped in
                this iteration. Defaults to { config.PPO_MAX_KL_DIV }.
            max_ppo_epochs (int, optional): Maximal number of SGD steps without
                exceeding KL divergence threshold.
                Defaults to { config.PPO_MAX_EPOCHS }.
            ppo_batch_size (int, optional): PPO batch size in SGD.
                Defaults to { config.PPO_BATCH_SIZE }.
            entropy_coef (float, optional): weight of entropy in the actor loss
                function. Defaults to { config.PPO_ENTROPY }.
            actor_lr (float, optional): Learning rate of the actor.
                Defaults to { config.A_LR }.
            critic_lr (float, optional): Learning rate of the critic.
                Defaults to { config.C_LR }.
            critic_num_target_updates (int, optional): Number of target q-value updates
                per iteration. Defaults to { config.NUM_TARGET_UPDATES }.
            num_critic_updates_per_target (int, optional): Number of gradient steps of
                critic for one target. Defaults to { config.NUM_CRITIC_UPDATES }.
            normalize_adv (bool, optional): Normalize advantages for actor.
                Defaults to { config.NORMALIZE_ADV }.
            obs_norm_alpha (float, optional): If set describes how much variance and
                mean should be updated each iteration. obs_norm_alpha == 1 means no
                update and obs_norm_alpha == 0 means replace old mean and std.
                Defaults to { config.NORM_ALPHA }
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
            max_frames (int, optional): Limit of frames for training. Defaults to
                { None }.
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
        self.ppo_epsilon = epsilon
        self.gae_lambda = gae_lambda
        self.kl_div_threshold = kl_div_threshold
        self.max_ppo_epochs = max_ppo_epochs
        self.ppo_batch_size = ppo_batch_size
        self.entropy_coef = entropy_coef
        self.kl_div_updates_counter = 0
        new_hparams = {
            "hparams/ppo_epsilon": self.ppo_epsilon,
            "hparams/gae_lambda": self.gae_lambda,
            "hparams/kl_div_threshold": self.kl_div_threshold,
            "hparams/max_ppo_epochs": self.max_ppo_epochs,
            "hparams/ppo_batch_size": self.ppo_batch_size,
        }
        self.hparams.update(new_hparams)

    def calculate_gae(
        self,
        obs: torch.tensor,
        next_obs: torch.tensor,
        ends: torch.tensor,
        done: torch.tensor,
        q_val: torch.tensor,
    ) -> torch.tensor:
        """
        Calculate advanatage using Generalized Advantage Estimation (GAE).

        Args:
            obs (torch.tensor): shape: [timesteps_no, obs_size]
            next_obs (torch.tensor): shape: [timesteps_no, obs_size]
            ends (torch.tensor): store 1 if a given timestamp was last in a batch,
                Shape: [timesteps_no]
            done (torch.tensor): store 1 if a given timestamp was terminating episode
                Shape: [timesteps_no]
            q_val (torch.tensor): tensor containing Q values for observations from
                the buffer. Shape: [timesteps_no]

        Returns:
            torch.tensor: tensor containing GAE advantages for observations from buffer
        """

        ###############################################################
        # You should implement Generalized Advantage Estimation
        # (https://arxiv.org/pdf/1506.02438.pdf).
        # You can do it in a following way:
        #   1. Calculate deltas for all observed states (see paper, between eq. 9 and 10.)
        #   2. Implement summing gae elements (starting from the last observations),
        #      taking care of the last element in a batch and the last element in
        #      a rollout. (see paper, eq. 16)
        # Hint: ends tensor store 1 if a given timestep was the last one because of the
        # environment steps limit. This means that for this timestep we need to estimate
        # state value using critic.
        # Below are some equations wich can be helpful:
        # End of your code #
        # A^1(t) = delta(t)
        # A^2(t) = delta(t) + gamma * delta(t+1)
        # A^3(t) = delta(t) + gamma * delta(t+1) + gamma^2 * delta(t+1)
        # GAE(t) = A^1(t) + lambda * A^2(t) + lambda^2  * A^3(t) + ...
        # GAE(t) = A^1(t) + lambda * GAE(t+1)
        ###############################################################

        # Your code is here #
        state_value = self.critic(obs)
        deltas = q_val - state_value.squeeze()

        advantage = torch.empty(size=deltas.size())

        discount = self.gae_lambda * self.gamma

        for i, delta in enumerate(reversed(deltas)):
            idx = -i - 1

            # Your code is here #
            if done[idx]:
                prev_state_value = 0  # type: float
            elif ends[idx]:
                prev_state_value = self.critic(next_obs[idx]).squeeze().item()
            gae_at_idx = prev_state_value * discount + delta
            # End of your code #
            advantage[idx] = gae_at_idx
            prev_state_value = gae_at_idx

        advantage = advantage.to(self.device)
        return advantage

    def calculate_advantage(self, buffer: Memory) -> torch.tensor:
        """
        Estimate advantage using GAE.

        Args:
            buffer (Memory): memory with samples

        Returns:
            torch.tensor: tensor containing GAE advantages for observations from buffer
        """
        obs = buffer.norm_obs
        next_obs = buffer.norm_next_obs

        reward = torch.tensor(buffer.rewards, dtype=torch.float32, device=self.device)
        done = torch.tensor(buffer.done, dtype=torch.float32, device=self.device)
        ends = torch.tensor(buffer.end, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            # use q_val and gae to obtain advantage
            # Your code is here #
            q_val = self.calculate_q_val(reward, done, next_obs)
            advantage = self.calculate_gae(obs, next_obs, ends, done, q_val)
            # End of your code #

        return advantage

    def _calculate_l_clip(
        self,
        action_logprobs: torch.tensor,
        new_logprobs: torch.tensor,
        advantages: torch.tensor,
    ):
        """
        Args:
            action_logprobs (torch.tensor): shape: [timesteps_no, actions_size]
            new_logprobs (torch.tensor): [timesteps_no, actions_size]
            advantages (torch.tensor): [timesteps_no]

        Returns:
            [type]: (torch.tensor): scalar value
        """
        ###############################################################
        # Implement L^CLIP loss (see PPO paper, eq. 7)
        # Hints:
        #   - Use torch.clamp function
        #   - Use self.ppo_epsilon
        ###############################################################
        # Your code is here #
        ratio = torch.exp(new_logprobs - action_logprobs)
        clipped_ratio = torch.clamp(ratio, 1 - self.ppo_epsilon, 1 + self.ppo_epsilon)
        l_clip = -(torch.min(ratio * advantages, clipped_ratio * advantages)).mean()
        # End of your code #
        assert l_clip.shape == torch.Size([])
        return l_clip


    def update_actor(self, advantages: torch.Tensor, buffer: Memory):
        f"""One iteration of actor update

        Args:
            advantages (torch.Tensor): advantages for observations from buffer
            buffer (Memory): buffer with samples
        """
        advantage_dataset = AdvantageDataset(advantages, buffer, self.normalize_adv)
        dataloader = DataLoader(
            advantage_dataset, batch_size=self.ppo_batch_size, shuffle=True
        )
        kl_div = 0.0
        self.loss["actor"] = 0
        self.loss["entropy"] = 0
        self.loss["sum"] = 0

        for i in range(self.max_ppo_epochs):
            if kl_div >= self.kl_div_threshold:
                break

            for advantages, action_logprobs, actions, norm_obs in dataloader:
                action_logprobs = action_logprobs.detach()
                # get distribution of actions for the current actor in given states
                new_dist = self.actor.get_actions_dist(norm_obs)
                # get log probabilities for taken actions for the current actor
                new_logprobs = new_dist.log_prob(torch.squeeze(actions))

                entropy = new_dist.entropy().mean()

                ###############################################################
                # Calculate actor loss
                # (see PPO paper, eq. 9 but skip the value function component)
                # Use self._clip_loss method and self.entropy_coef parameter
                ###############################################################
                # Your code is here #
                l_clip = self._calculate_l_clip(
                    action_logprobs, new_logprobs, advantages
                )
                loss = l_clip - self.entropy_coef * entropy
                # End of your code #

                self.actor_optimizer.zero_grad()
                loss.backward()
                self.actor_optimizer.step()
                self.loss["actor"] += l_clip.item()
                self.loss["entropy"] += entropy.item()
                self.loss["sum"] += loss.item()

            kl_div = utils.kl_divergence(action_logprobs.cpu(), new_logprobs.cpu())

        logger.debug(f"PPO update finished after {i} epochs with KL = {kl_div}")
        self.kl_div_updates_counter += i + 1


    def add_tensorboard_logs(self, *args, **kwargs):
        super().add_tensorboard_logs(*args, **kwargs)
        if self.debug_mode:
            self.log_ppo_epochs()

    def log_ppo_epochs(self):
        if self.iteration == 0:
            mean_updates = self.kl_div_updates_counter
        elif self.iteration % self.stats_freq == 0:
            mean_updates = self.kl_div_updates_counter / self.stats_freq
        else:
            denominator = self.iteration % self.stats_freq
            mean_updates = self.kl_div_updates_counter / denominator

        self.tensorboard_writer.log_kl_div_updates(
            self.iteration,
            self.stats_logger.frames,
            self.stats_logger.rollouts,
            mean_updates,
        )
        self.kl_div_updates_counter = 0


if __name__ == "__main__":
    model = PPO(
        env_name="CartPole-v0",
        # gamma=0.95,
        # actor_lr=3e-3,
        # critic_lr=3e-4,
        # batch_size=200,
        # ppo_batch_size=128,
        kl_div_threshold=0.15,
        stats_freq=5,
        max_ppo_epochs=10,
        tensorboard_dir="logs",
    )
    model.train()
