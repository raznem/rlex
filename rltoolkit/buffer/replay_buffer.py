import numpy as np
import torch
from rltoolkit.buffer.memory import Memory, MemoryMeta


class MetaReplayBuffer(MemoryMeta):
    def __init__(
        self,
        size: int,
        obs_shape,
        obs_norm: bool = False,
        dtype: torch.dtype = torch.float32,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.size = size
        self.obs_shape = obs_shape
        self.obs_norm = obs_norm
        self.dtype = dtype

        self.obs_idx = None
        self.ts_idx = None
        self.current_len = None
        self.reset_idx()

        self._obs = np.empty([self.size, self.obs_shape])
        self._obs_idx = np.empty(self.size, dtype=np.int)
        self._next_obs_idx = np.empty(self.size, dtype=np.int)

    def reset_idx(self):
        self.obs_idx = 0
        self.ts_idx = 0
        self.current_len = 0

    @property
    def obs(self):
        return self._obs[self._obs_idx[: self.current_len]]

    @property
    def norm_obs(self):
        return self.normalize(self.obs)

    @property
    def next_obs(self):
        return self._obs[self._next_obs_idx[: self.current_len]]

    @property
    def norm_next_obs(self):
        return self.normalize(self.next_obs)

    def __len__(self):
        return self.current_len

    def add_obs(self, obs: torch.Tensor):
        self._obs[self.obs_idx] = obs.cpu()
        obs_idx = self.obs_idx
        self.obs_idx = (self.obs_idx + 1) % self.size
        return obs_idx

    def addition(*args, **kwargs):
        pass

    def add_timestep(self, obs_idx: int, next_obs_idx: int, *args, **kwargs):
        self._obs_idx[self.ts_idx] = obs_idx
        self._next_obs_idx[self.ts_idx] = next_obs_idx
        self.addition(*args, **kwargs)

        if next_obs_idx < self.ts_idx:
            self.current_len = self.ts_idx + 1
            self.ts_idx = 0
        else:
            self.ts_idx += 1
        self.current_len = max(self.ts_idx, self.current_len)

    def normalize(self, obs: torch.tensor) -> torch.tensor:
        if self.obs_norm:
            return super().normalize(obs)
        else:
            return obs

    def update_obs_mean_std(self):
        if len(self.obs) > 10:
            self.obs_mean = torch.tensor(self.obs.mean(axis=0), dtype=self.dtype)
            self.obs_std = torch.tensor(self.obs.std(axis=0), dtype=self.dtype)


class ReplayBuffer(MetaReplayBuffer):
    def __init__(
        self, size: int, obs_shape, act_shape, discrete: bool = False, *args, **kwargs
    ):
        super().__init__(size, obs_shape, *args, **kwargs)
        self.discrete = discrete
        if self.discrete:
            self._actions = np.empty(self.size, dtype=np.int)
        else:
            self._actions = np.empty([self.size, act_shape])
        self._rewards = np.empty(self.size, dtype=np.float32)
        self._done = np.empty(self.size, dtype=np.bool_)
        self._end = np.empty(self.size, dtype=np.bool_)  # only for episode extraction

        if self.obs_std is None and self.obs_mean is None:
            self.obs_mean = torch.zeros(obs_shape, device=self.device)
            self.obs_std = torch.ones(obs_shape, device=self.device)

    @property
    def actions(self):
        return self._actions[: self.current_len]

    @property
    def rewards(self):
        return self._rewards[: self.current_len]

    @property
    def done(self):
        return self._done[: self.current_len]

    @property
    def end(self):
        return self._end[: self.current_len]

    def addition(self, action, rew, done, end):
        self._actions[self.ts_idx] = action.cpu()
        self._rewards[self.ts_idx] = rew
        self._done[self.ts_idx] = done
        self._end[self.ts_idx] = end

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
        raise DeprecationWarning
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

    def make_obs_memory_tensor(self, obs):
        processed = torch.tensor(obs, dtype=self.dtype, device=self.device)
        processed = processed.reshape(-1, self.obs_shape)
        return processed

    def last_end(self, idx):
        end = self._end[idx]
        while not end:
            idx -= 1
            if idx < 0:
                idx = self.current_len - 1
            end = self._end[idx]
        return idx

    def last_rollout(self):
        """Get elements of last full rollout:
            1. Find last end
            2. Grab elements till the next end

        Returns:
            Memory: memory with last rollout only
        """
        i = self.last_end(self.ts_idx - 1)
        next_end = False

        obs = []
        next_obs = []
        actions = []
        rewards = []
        dones = []
        while not next_end:
            new_obs = self.make_obs_memory_tensor(self._obs[self._obs_idx[i]])
            new_next_obs = self.make_obs_memory_tensor(self._obs[self._next_obs_idx[i]])
            obs.insert(0, new_obs)
            next_obs.insert(0, new_next_obs)
            actions.insert(0, self._actions[i])
            rewards.insert(0, self._rewards[i])
            dones.insert(0, self._end[i])
            i -= 1
            if i < 0:
                i = self.current_len - 1
            next_end = self._end[i]

        memory = Memory()
        memory.add_rollout(
            obs=obs, actions=actions, action_logprobs=[], rewards=rewards, dones=dones
        )
        return memory

    def sample_batch(self, batch_size=64, device=None):
        f"""Sample batch of tensors from buffer

        Args:
            batch_size (int, optional): batch size. Defaults to { 64 }.
            device (torch.device, optional): use GPU/CPU device. Defaults to { None }.

        Returns:
            list: list of elements from buffer (obs, next_obs, actions, rewards, dones)
        """
        idxs = np.random.randint(0, self.__len__(), batch_size)

        if self.discrete:
            actions_type = torch.long
        else:
            actions_type = self.dtype

        obs = torch.as_tensor(
            self._obs[self._obs_idx[idxs]], dtype=self.dtype, device=self.device
        )
        next_obs = torch.as_tensor(
            self._obs[self._next_obs_idx[idxs]], dtype=self.dtype, device=self.device
        )
        if self.obs_norm:
            obs = self.normalize(obs)
            next_obs = self.normalize(next_obs)

        batch = [
            obs,
            next_obs,
            torch.as_tensor(self._actions[idxs], dtype=actions_type),
            torch.as_tensor(self._rewards[idxs], dtype=actions_type),
            torch.as_tensor(self._done[idxs], dtype=torch.int8),
        ]

        if device:
            batch = [item.to(device) for item in batch]

        return batch
