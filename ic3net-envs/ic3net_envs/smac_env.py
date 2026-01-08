#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SMAC (StarCraft Multi-Agent Challenge) environment wrapper for IC3Net.

Adapts SMAC's MultiAgentEnv interface to IC3Net's expected Gym interface.
"""

import gym
import numpy as np
from gym import spaces


class SMACEnv(gym.Env):

    def __init__(self):
        self.__version__ = "0.0.1"
        self.smac_env = None

        # Set placeholder spaces to satisfy gym's environment checker
        # These will be properly initialized in multi_agent_init()
        self.action_space = spaces.Discrete(1)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(1,), dtype=np.float32
        )

    def init_args(self, parser):
        """Add SMAC-specific arguments to parser."""
        env = parser.add_argument_group('SMAC task')
        env.add_argument('--map_name', type=str, default='3m',
                         help="SMAC map name (e.g., '3m', '8m', '2s3z')")
        env.add_argument('--difficulty', type=str, default='7',
                         help="Difficulty level: 1-10 or 'A' for very easy to very hard")
        env.add_argument('--step_mul', type=int, default=8,
                         help="StarCraft step multiplier (frames per action)")
        env.add_argument('--reward_scale_rate', type=float, default=20,
                         help="Reward scaling factor")

    def multi_agent_init(self, args):
        """Initialize SMAC environment with parsed arguments."""
        from smac.env import StarCraft2Env

        # Create SMAC environment
        self.smac_env = StarCraft2Env(
            map_name=args.map_name,
            difficulty=args.difficulty,
            step_mul=args.step_mul,
            reward_scale_rate=args.reward_scale_rate
        )

        # Get environment info
        env_info = self.smac_env.get_env_info()
        self.n_agents = env_info['n_agents']
        self.episode_limit = env_info['episode_limit']
        self.obs_size = env_info['obs_shape']
        self.n_actions = env_info['n_actions']

        # Set action space (MultiDiscrete with n_actions per agent)
        # Note: GymWrapper handles multi-agent, so we define action space per agent
        self.action_space = spaces.MultiDiscrete([self.n_actions])

        # Set observation space (Box with obs_size, using local observations only)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.obs_size,), dtype=np.float32
        )

        # Initialize stat dict
        self.stat = {}

    def reset(self):
        """Reset the environment and return initial observations."""
        # SMAC returns (obs, state) but we only use obs
        self.smac_env.reset()

        # Get observations (list of length n_agents)
        obs = self.smac_env.get_obs()

        # Convert to numpy array (shape: [n_agents, obs_size])
        obs = np.array(obs)

        # Get available actions for all agents (shape: [n_agents, n_actions])
        avail_actions = self.smac_env.get_avail_actions()
        self.last_avail_actions = np.array(avail_actions)

        # Initialize episode stats
        self.stat = {}

        return obs

    def step(self, action):
        """
        Take a step in the environment.

        Parameters
        ----------
        action : list/ndarray of agent actions, or list of action dimensions

        Returns
        -------
        obs, reward, terminated, info : tuple
            obs (ndarray): observations for all agents
            reward (ndarray): rewards for all agents
            terminated (bool): whether episode is done
            info (dict): diagnostic information
        """
        # Handle multiple action dimensions (e.g., with hard_attn communication)
        # If action is a list of lists/arrays, take the first dimension (main actions)
        if isinstance(action, (list, tuple)) and len(action) > 0:
            if isinstance(action[0], (list, np.ndarray)):
                # Multiple action dimensions: take first (main actions)
                action = action[0]

        # Convert action to list if needed
        action = np.atleast_1d(np.array(action).squeeze()).tolist()

        # SMAC step returns (reward, terminated, info) without observations
        reward, terminated, info = self.smac_env.step(action)

        # Get new observations separately
        obs = self.smac_env.get_obs()

        # Convert to numpy array
        obs = np.array(obs)

        # Get available actions for all agents (shape: [n_agents, n_actions])
        avail_actions = self.smac_env.get_avail_actions()
        self.last_avail_actions = np.array(avail_actions)

        # Add available actions to info dict for action masking
        info['avail_actions'] = self.last_avail_actions

        # Broadcast scalar reward to all agents
        reward = np.full(self.n_agents, reward)

        # Update stats on episode end
        if terminated:
            self.stat['success'] = info.get('battle_won', 0)

        return obs, reward, terminated, info

    def get_avail_actions(self):
        """Return available actions for all agents."""
        if hasattr(self, 'last_avail_actions'):
            return self.last_avail_actions
        else:
            # If not yet initialized, get from SMAC
            avail_actions = self.smac_env.get_avail_actions()
            return np.array(avail_actions)

    def get_stat(self):
        """Return episode statistics."""
        return self.stat

    def reward_terminal(self):
        """Return terminal rewards (SMAC handles this internally)."""
        return np.zeros(self.n_agents)

    def close(self):
        """Cleanup SMAC environment."""
        if self.smac_env is not None:
            self.smac_env.close()

    def seed(self, seed=None):
        """Set random seed."""
        if self.smac_env is not None:
            self.smac_env.seed = seed
