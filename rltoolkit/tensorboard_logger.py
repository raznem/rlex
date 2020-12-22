import ctypes
import logging
import multiprocessing as mp
import numbers
from os import path
from typing import Any, Dict, Iterable, Tuple

import gym
import numpy as np
import torch
from pyvirtualdisplay import Display
from torch.utils.tensorboard import SummaryWriter

from rltoolkit.buffer import Memory
from rltoolkit.utils import measure_time

logger = logging.getLogger(__name__)


class TensorboardWriter(SummaryWriter):
    """
    Custom SummaryWriter class for RL purposes.
    """

    def __init__(
        self,
        env_name: str,
        log_dir: str,
        filename: str,
        render: bool = True,
        fps: int = 24,
        frames_count: int = 2000,
    ):
        """
        Arguments:
            env_name {str} -- name of env used for getting rendering resolution
            log_dir {str} -- directory to store tensorboard logs

        Keyword Arguments:
            filename {str} -- filename of the experiment
            fps {int} -- number of frames per seconds. Used in recording rollout.
                (default: {24})
            frames_count {int} -- maximal length of recorded rollout (default: {2000})
        """
        self.render = render
        if self.render:
            self.fps = fps
            self.recording_process = None
            self.rendering_video_index = 0

            self.frames_shape = self.get_rendering_shape(env_name, frames_count)
            self.array_size = int(np.prod(self.frames_shape))
            self.shared_video_frames = mp.Array(ctypes.c_uint8, self.array_size)

        log_dir = path.join(log_dir, filename)
        super().__init__(log_dir=log_dir)

    def get_rendering_shape(
        self, env_name: str, frames_count: int
    ) -> Tuple[int, int, int, int]:
        """
        Create dummy environment instance to get rendering resolution.
        Due to classical controll environments initialization it has to be done in the
        separate process.

        Arguments:
            env_name {str} -- name of the environment
            frames_count {int} -- maximal length of video in frames

        Returns:
            Tuple[int, int, int, int] -- frames_count, height, width, and chanels of
                a video
        """
        resolution = mp.Array(ctypes.c_uint16, 3)

        ctx = mp.get_context("spawn")
        p = ctx.Process(target=get_resolution_mp, args=(env_name, resolution))

        p.start()
        p.join()

        height, width, chanels = np.frombuffer(resolution.get_obj(), dtype=np.uint16)
        height, width, chanels = int(height), int(width), int(chanels)

        return frames_count, height, width, chanels

    @measure_time
    def record_episode(self, a2c: Any, i: int, done: bool = False):
        """
        At the first run time just start collecting frames in separete process.
        At each next first join previously started process and send video to the
        tensorboard, after that start new collecting frames process.
        At last run (flag done=True) do above and wait for the last process and send
        video to the tensorboard

        Args:
            a2c (A2C): a2c object with the agent.
            i (int): iteration number - used to log this in the tensorboard
            done (bool, optional): information if recording is done at the end of the
                experiment. Defaults to False.
        """
        if not self.render:
            return

        if self.recording_process is not None:
            self._join_process_and_add_video()

        self.rendering_video_index = i
        self._start_collecting_rollout_frames(a2c)

        if done:
            self._join_process_and_add_video()

    def _join_process_and_add_video(self):
        """
        Join recording process and send generated frames to the tensorboard.
        """
        self.recording_process.join()
        frames = self._from_mp_array_to_tensor(self.shared_video_frames)
        self.add_video("Episode", frames, self.rendering_video_index, fps=self.fps)
        self.shared_video_frames = mp.Array(ctypes.c_uint8, self.array_size)

    def _start_collecting_rollout_frames(self, a2c: Any):
        """
        Start collecting frames from a rollout in the different process.

        Args:
            a2c (A2C): a2c object with the agent.
        """
        args = (a2c, self.shared_video_frames, self.frames_shape)
        self.recording_process = mp.Process(target=_record_episode_mp, args=args)
        self.recording_process.start()

    def _from_mp_array_to_tensor(self, mp_arr: mp.Array) -> torch.tensor:
        """
        Convert multiprocessing.Array with recorded frames into torch.tensor.
        Additionally remove last black frames and change chanel position.
        Original shape: B x H x W x C
        Output shape    B x C x H x W

        Args:
            mp_arr (mp.Array): recorded frames in the shared memory.

        Returns:
            torch.tensor: tensor with data ready to use in tensorboard writer
        """
        arr = np.frombuffer(mp_arr.get_obj(), dtype=np.uint8)
        arr = arr.reshape(self.frames_shape)
        arr = self._remove_black_frames(arr)
        arr = torch.tensor(arr)
        arr = torch.unsqueeze(arr.permute(0, 3, 1, 2), dim=0)
        return arr

    def _remove_black_frames(self, arr: np.array) -> np.array:
        """
        Not always rollout will last for full buffer capacity so we need to remove all
        black frames.

        Args:
            arr (np.array): array with frames.

        Returns:
            np.array: array with frames without black frames at the end.
        """
        for frame in range(self.frames_shape[0]):
            if (arr[frame] == 0).all():
                sl = slice(frame, None)
                arr = np.delete(arr, sl, 0)
                break
        return arr

    def log_returns(self, i: int, buffer: Memory):
        assert buffer.rewards and buffer.done

        self.add_histogram("Return", buffer.returns_rollouts, i)

    def log_kl_div_updates(
        self, iterations: int, frames: int, rollouts: int, updates_no: float
    ):
        self.add_scalar("PPO/KL_updates_mean/per_iterations", updates_no, iterations)
        self.add_scalar("PPO/KL_updates_mean/per_frames", updates_no, frames)
        self.add_scalar("PPO/KL_updates_mean/per_rollouts", updates_no, rollouts)

    def log_sac_alpha(self, iterations: int, alpha: float):
        self.add_scalar("SAC/Alpha_per_iterations", alpha, iterations)

    def log_actions(self, i: int, buffer: Memory):
        assert len(buffer.actions) > 0
        actions = self._get_buffer_elem_tensor(buffer.actions)

        if len(actions.shape) == 1:
            self._add_single_variable_histogram("Action", actions, i)
        elif len(actions.shape) == 2:
            self._add_multiple_variables_histograms("Action", actions, i)
        else:
            raise ValueError("2D actions are not supported")

    @staticmethod
    def _get_buffer_elem_tensor(buffer_list: Iterable) -> torch.tensor:
        first_val = buffer_list[0]
        if isinstance(first_val, numbers.Number):
            output = torch.tensor(buffer_list)
        elif isinstance(first_val, np.ndarray):
            output = np.array(buffer_list)
        elif isinstance(first_val, torch.Tensor):
            output = torch.cat(buffer_list)
        else:
            raise TypeError("Unsupported action type.")
        return output

    def _add_single_variable_histogram(self, name: str, vector: torch.tensor, i: int):
        self.add_histogram(f"{name}/0", vector, i)

    def _add_multiple_variables_histograms(
        self, name: str, matrix: torch.tensor, i: int
    ):
        matrix_rows = matrix.shape[1]
        for j in range(matrix_rows):
            var_j = matrix[:, j]
            self.add_histogram(f"{name}/{j}", var_j, i)

    def log_observations(self, i: int, buffer: Memory):
        assert len(buffer) > 0
        if isinstance(buffer.obs[0], torch.Tensor):
            obs = buffer.obs.squeeze()
        elif isinstance(buffer.obs[0], np.ndarray):
            obs = np.stack(buffer.obs, axis=0)
        else:
            raise TypeError("Unsupported observation type.")

        if len(obs.shape) == 1:
            self._add_single_variable_histogram("Observation", obs, i)
        elif len(obs.shape) == 2:
            self._add_multiple_variables_histograms("Observation", obs, i)
        else:
            raise ValueError("2D observations are not supported")

    def log_running_return(
        self, iterations: int, frames: int, rollouts: int, running_return: float
    ):
        self.add_scalar("1_Running_return/per_iterations", running_return, iterations)
        self.add_scalar("1_Running_return/per_frames", running_return, frames)
        self.add_scalar("1_Running_return/per_rollouts", running_return, rollouts)

    def log_test_return(
        self, iterations: int, frames: int, rollouts: int, test_return: float
    ):
        self.add_scalar("1_Test_return/per_iterations", test_return, iterations)
        self.add_scalar("1_Test_return/per_frames", test_return, frames)
        self.add_scalar("1_Test_return/per_rollouts", test_return, rollouts)

    def log_loss(self, i: int, loss: Dict[str, int]):
        for key, value in loss.items():
            label = "Loss/" + key.capitalize()
            self.add_scalar(label, value, i)

    def log_hyperparameters(self, hparam_dict, metric_dict):
        """
        [summary]

        Args:
            hyps (dict): [description]
            metrics (dict): [description]
        """
        self.add_hparams(hparam_dict, metric_dict)

    def log_obs_mean_std(self, iterations: int, mean: torch.tensor, std: torch.tensor):
        """
        Log all observation means and standard deviations.
        """

        for i in range(len(mean)):
            self.add_scalar(f"Obs/mean/{i}", mean[i], iterations)
            self.add_scalar(f"Obs/std/{i}", std[i], iterations)


def _record_episode_mp(
    a2c: Any, mp_arr: mp.Array, frames_shape: Tuple[int, int, int, int]
):
    frames_count = frames_shape[0]
    with Display():
        done = False
        obs = a2c.env.reset()
        obs = a2c.process_obs(obs)
        i = 0

        frames = np.frombuffer(mp_arr.get_obj(), dtype=np.uint8).reshape(frames_shape)

        frames[i] = a2c.env.render(mode="rgb_array")
        while not done:
            if i == frames_count:
                logger.warning("Too small frame count declared for full video.")
                break
            action, action_logprobs = a2c.actor.act(obs)
            action_proc = a2c.process_action(action, obs)
            obs, rew, done, _ = a2c.env.step(action_proc)
            obs = a2c.process_obs(obs)
            frames[i] = a2c.env.render(mode="rgb_array")
            i += 1

    return 0


def get_resolution_mp(env_name: str, mp_resolution: mp.Array):
    with Display():
        resolution = np.frombuffer(mp_resolution.get_obj(), dtype=np.uint16)
        env = gym.make(env_name)
        env.reset()
        frame = env.render(mode="rgb_array")
        env.close()
    for i in range(len(frame.shape)):
        resolution[i] = frame.shape[i]
