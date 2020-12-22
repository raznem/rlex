import numpy as np
import torch
from typing import Iterable


class Memory:
    def __init__(self):
        self._obs = []
        self._next_obs_idx = []
        self._actions = []
        self._action_logprobs = []
        self._rewards = []
        self._done = []
        self._new_rollout_idx = []  # For obs skip previous, for next_obs skip this
        self.current_len = 0

    @property
    def obs(self):
        obs = []
        rollout = 0
        for i in range(self.current_len):
            if i != self._new_rollout_idx[rollout] - 1:
                obs.append(self._obs[i])
            else:
                if rollout < len(self._new_rollout_idx) - 1:
                    rollout += 1
        return obs

    @property
    def next_obs(self):
        obs = []
        rollout = 0
        for i in range(1, self.current_len):
            if i != self._new_rollout_idx[rollout]:
                obs.append(self._obs[i])
            else:
                if rollout < len(self._new_rollout_idx) - 1:
                    rollout += 1
        return obs

    @property
    def actions(self):
        return self._actions

    @property
    def action_logprobs(self):
        return self._action_logprobs

    @property
    def rewards(self):
        return self._rewards

    @property
    def done(self):
        return self._done

    @property
    def average_returns_per_rollout(self):
        return sum(self.returns_rollouts) / self.rollouts_no

    @property
    def returns_rollouts(self) -> np.array:
        returns = []
        return_ = 0

        for i, done in enumerate(self.done):
            return_ += self.rewards[i]
            if done:
                returns.append(return_)
                return_ = 0
        return np.array(returns)

    @property
    def rollouts_no(self) -> int:
        return sum(self._done)

    def add_obs(self, obs):
        self._obs.append(obs)
        obs_idx = len(self._obs) - 1
        self.current_len += 1
        return obs_idx

    def add_timestep(
        self, obs_idx, next_obs_idx, action, action_logprobs, reward, done
    ):
        self._next_obs_idx.append(next_obs_idx)
        self._actions.append(action)
        self._action_logprobs.append(action_logprobs)
        self._rewards.append(reward)
        self._done.append(done)

    def __len__(self):
        return len(self._next_obs_idx)

    def __getitem__(self, obs_idx):
        next_obs_idx = self._next_obs_idx[obs_idx]
        return (
            self._obs[obs_idx],
            self._obs[next_obs_idx],
            self._actions[obs_idx],
            self._action_logprobs[obs_idx],
            self._rewards[obs_idx],
            self._done[obs_idx],
        )

    def new_rollout(self):
        if len(self._obs) > 0:
            self._new_rollout_idx.append(self.current_len)

    def end_rollout(self):
        self._new_rollout_idx.append(self.current_len)

    def add_rollout(self, obs, actions, action_logprobs, rewards, dones):
        len_before = self.__len__()
        self._obs += obs
        self._actions += actions
        self._action_logprobs += action_logprobs
        self._rewards += rewards
        self._done += dones

        new_next_idx = list(range(len_before + 1, len(self._obs)))
        self._next_obs_idx.append(new_next_idx)
        self.current_len += len(obs)
        self.end_rollout()

    def add_buffers_list(self, buffer_list):
        for new_buffer in buffer_list:
            self._obs += new_buffer._obs
            self._actions += new_buffer._actions
            self._action_logprobs += new_buffer._action_logprobs
            self._rewards += new_buffer._rewards
            self._done += new_buffer._done

            last_idx = 0 if len(self._next_obs_idx) == 0 else self._next_obs_idx[-1]
            for i in new_buffer._next_obs_idx:
                self._next_obs_idx.append(last_idx + i + 1)


class MetaReplayBuffer:
    def __init__(self, size, obs_shape):
        self.size = size
        self.obs_idx = 0
        self.ts_idx = 0
        self.current_len = 0

        self._obs = np.empty([self.size, obs_shape])
        self._obs_idx = np.empty(self.size, dtype=np.int)

    @property
    def obs(self):
        return self._obs[self._obs_idx[: self.current_len]]

    def __len__(self):
        return self.current_len

    def add_obs(self, obs: torch.Tensor):
        self._obs[self.obs_idx] = obs.cpu()
        obs_idx = self.obs_idx
        self.obs_idx = (self.obs_idx + 1) % self.size
        return obs_idx


class ReplayBuffer(MetaReplayBuffer):
    def __init__(
        self, size, obs_shape, act_shape, dtype=torch.float32, discrete: bool = False
    ):
        super().__init__(size, obs_shape)
        self._next_obs_idx = np.empty(self.size, dtype=np.int)
        self.discrete = discrete
        if self.discrete:
            self._actions = np.empty(self.size, dtype=np.int)
        else:
            self._actions = np.empty([self.size, act_shape])
        self._rewards = np.empty(self.size, dtype=np.float32)
        self._done = np.empty(self.size, dtype=np.bool_)
        self.dtype = dtype

    @property
    def next_obs(self):
        return self._obs[self._next_obs_idx[: self.current_len]]

    @property
    def actions(self):
        return self._actions[: self.current_len]

    @property
    def rewards(self):
        return self._rewards[: self.current_len]

    @property
    def done(self):
        return self._done[: self.current_len]

    def add_timestep(
        self, obs_idx: int, next_obs_idx: int, action: Iterable, rew: float, done: bool
    ):
        if obs_idx < self.ts_idx:
            self.current_len = self.ts_idx
            self.ts_idx = 0

        self._obs_idx[self.ts_idx] = obs_idx
        self._next_obs_idx[self.ts_idx] = next_obs_idx
        self._actions[self.ts_idx] = action
        self._rewards[self.ts_idx] = rew
        self._done[self.ts_idx] = done
        self.ts_idx += 1
        self.current_len = max(self.ts_idx, self.current_len)

    def __getitem__(self, idx):
        assert idx < self.current_len, IndexError
        return (
            self._obs[self._obs_idx[idx]],
            self._obs[self._next_obs_idx[idx]],
            self._actions[idx],
            self._rewards[idx],
            self._done[idx],
        )

    def add_buffer(self, memory: Memory):
        obs = memory._obs
        new_rollout_idx = memory._new_rollout_idx
        i = 0
        obs_idx = self.add_obs(obs[i])
        timesteps = zip(memory.actions, memory.rewards, memory.done)
        for action, rew, done in timesteps:
            i += 1
            next_idx = self.add_obs(obs[i])
            if i in new_rollout_idx:
                i += 1
                continue
            self.add_timestep(obs_idx, next_idx, action, rew, done)
            obs_idx = next_idx

    def last_done(self, idx):
        done = self._done[idx]
        while not done:
            idx -= 1
            if idx < 0:
                idx = self.current_len - 1
            done = self._done[idx]
        return idx

    def last_rollout(self):
        """Get elements of last full rollout:
            1. Find last done
            2. Grab elements till the next done

        Returns:
            Memory: memory with last rollout only
        """
        i = self.last_done(self.ts_idx - 1)
        next_done = False

        obs = []
        next_obs = []
        actions = []
        rewards = []
        dones = []
        while not next_done:
            obs.insert(0, self._obs[self._obs_idx[i]])
            next_obs.insert(0, self._obs[self._next_obs_idx[i]])
            actions.insert(0, self._actions[i])
            rewards.insert(0, self._rewards[i])
            dones.insert(0, self._done[i])
            i -= 1
            if i < 0:
                i = self.current_len - 1
            next_done = self._done[i]

        memory = Memory()
        memory.add_rollout(
            obs=obs, actions=actions, action_logprobs=[], rewards=rewards, dones=dones
        )
        return memory

    def sample_batch(self, batch_size=64):
        f"""Sample batch of tensors from buffer

        Args:
            batch_size (int, optional): batch size. Defaults to { 64 }.

        Returns:
            list: list of elements from buffer
        """
        idxs = np.random.randint(0, self.__len__(), batch_size)

        if self.discrete:
            actions_type = torch.long
        else:
            actions_type = self.dtype

        batch = [
            torch.as_tensor(self._obs[self._obs_idx[idxs]], dtype=self.dtype),
            torch.as_tensor(self._obs[self._next_obs_idx[idxs]], dtype=self.dtype),
            torch.as_tensor(self._actions[idxs], dtype=actions_type),
            torch.as_tensor(self._rewards[idxs], dtype=actions_type),
            torch.as_tensor(self._done[idxs], dtype=torch.int8),
        ]

        return batch

