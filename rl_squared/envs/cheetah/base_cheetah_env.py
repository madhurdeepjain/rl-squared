from typing import Tuple, Any
from abc import ABC

import gymnasium as gym

from gymnasium.envs.mujoco.half_cheetah_v5 import HalfCheetahEnv as HalfCheetahEnv_
from rl_squared.envs.base_mujoco_meta_env import BaseMujocoMetaEnv


class BaseCheetahEnv(HalfCheetahEnv_, BaseMujocoMetaEnv, ABC):
    def __init__(self):
        """
        Initialize the Mujoco Ant environment for meta-learning.
        """
        BaseMujocoMetaEnv.__init__(self)
        HalfCheetahEnv_.__init__(self)
        pass

    def get_spaces(self) -> Tuple[gym.Space, gym.Space]:
        """
        Returns the observation space and the action space.

        Returns:
            Tuple
        """
        return self.observation_space, self.action_space

    @property
    def observation_space(self) -> gym.Space:
        """
        Returns the observation space for the environment.

        Returns:
          gym.Space
        """
        return self._observation_space

    @observation_space.setter
    def observation_space(self, value: Any) -> None:
        """
        Set the observation space for the environment.

        Returns:
          gym.Space
        """
        self._observation_space = value

    @property
    def action_space(self) -> gym.Space:
        """
        Set the action space for the environment.

        Returns:
            gym.Space
        """
        return self._action_space

    @action_space.setter
    def action_space(self, value: Any) -> None:
        """
        Set the action space for the environment.

        Returns:
            gym.Space
        """
        self._action_space = value

    def viewer_setup(self) -> None:
        """
        Set up the viewer for rendering the environment.

        Returns:
            None
        """
        pass

    def render(self):
        return super().render()
