# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import copy
from tabnanny import process_tokens
from tracemalloc import start

import numpy as np
import torch
from gymnasium import spaces
from isaacgym import gymtorch
from isaacgym.torch_utils import to_torch


# VecEnv Wrapper for ShadowHand
class ShadowHandMultiVecTask:
    def __init__(self, task, rl_device, clip_observations=5.0, clip_actions=1.0):
        self.task = task

        self.num_environments = task.num_envs
        self.num_states = task.num_states
        self.num_actions = task.num_actions
        self.num_hand_obs = task.num_hand_obs

        self.num_observations = task.num_obs - self.num_hand_obs
        self.nums_share_observations = self.num_observations + self.num_hand_obs
        self.agent_index = self.task.agent_index
        self.num_agents = len(self.agent_index[0] * 2)  # used for multi-agent environments

        self.clip_obs = clip_observations
        self.clip_actions = clip_actions
        self.rl_device = rl_device

        print('RL device: ', rl_device)

        # COMPATIBILITY
        # self.observation_space = [Box(low=np.array([-10]*self.n_agents), high=np.array([10]*self.n_agents)) for _ in range(self.n_agents)]
        self.obs_space = [
            spaces.Box(low=-np.Inf, high=np.Inf, shape=(self.num_observations,))
            for _ in range(self.num_agents)
        ]
        self.share_observation_space = [
            spaces.Box(low=-np.Inf, high=np.Inf, shape=(self.nums_share_observations,))
            for _ in range(self.num_agents)
        ]
        '''
        self.hand_dof_index_dict = {
            "WRJ": [0, 1],
            "FFJ": [2, 3, 4, 5],
            "MFJ": [6, 7, 8, 9],
            "RFJ": [10, 11, 12, 13],
            "LFJ": [14, 15, 16, 17, 18],
            "THJ": [19, 20, 21, 22, 23],
        }
        actuated_dof_indices: [ 0,  1,  2,  3,  4,  6,  7,  8, 10, 11, 12, 14, 15, 16, 17, 19, 20, 21, 22, 23]
        '''

        self.hand_dof_index_dict = [
            [0, 1],
            [2, 3, 4],
            [6, 7, 8, 9],
            [10, 11, 12, 13],
            [14, 15, 16, 17, 18],
            [19, 20, 21, 22, 23],
        ]
        self.hand_actuated_dof_index_dict = [
            [0, 1],
            [2, 3, 4],
            [6, 7, 8],
            [10, 11, 12],
            [14, 15, 16, 17],
            [19, 20, 21, 22, 23],
        ]
        if self.task.num_actions == 26:
            self.hand_actuated_dof_index_dict = [
                [-1, -1, -1, -1, -1, -1, 0, 1],
                [2, 3, 4],
                [6, 7, 8],
                [10, 11, 12],
                [14, 15, 16, 17],
                [19, 20, 21, 22, 23],
            ]

        self.agent_dof_index = []
        self.agent_actuated_dof_index = []
        self.agent_finger_index = [[] for _ in range(len(self.agent_index[0]))]

        for hand_num in range(len(self.agent_index)):
            for i, agent in enumerate(self.agent_index[hand_num]):
                temp1_index = copy.deepcopy(self.hand_dof_index_dict[agent[0]])
                temp2_index = copy.deepcopy(self.hand_actuated_dof_index_dict[agent[0]])

                # a agent
                for j in agent[1:]:
                    temp1_index += self.hand_dof_index_dict[j]
                    temp2_index += self.hand_actuated_dof_index_dict[j]
                    # if j != 0:
                    #     self.agent_finger_index[i].append(j - 1)
                self.agent_dof_index.append(temp1_index)
                self.agent_actuated_dof_index.append(temp2_index)

        print('self.agent_dof_index: ', self.agent_dof_index)
        print('self.hand_actuated_dof_index_dict: ', self.agent_actuated_dof_index)

        self.act_space = tuple(
            [
                spaces.Box(
                    low=np.ones(len(agent_dof_index)) * -clip_actions,
                    high=np.ones(len(agent_dof_index)) * clip_actions,
                )
                for agent_dof_index in self.agent_actuated_dof_index
            ]
        )

    # def process_sub_agent_obs(self, agent_dof_index, agent_finger_index, obs):
    #     sub_agent_obs = []
    #     for i in range(len(agent_dof_index)):
    #         sub_dof_pos = obs[:, agent_dof_index[i]]
    #         sub_dof_vel = obs[:, [index + 24 for index in agent_dof_index[i]]]
    #         sub_dof_force = obs[:, [index + 48 for index in agent_dof_index[i]]]
    #         if i != 0:
    #             sub_finger_state = obs[:, 72:72+65].reshape(self.num_envs, 5, 13)[:, agent_finger_index[i-1], :].reshape(self.num_envs, -1)
    #             sub_finger_force = obs[:, 72+65:72+65+30].reshape(self.num_envs, 5, 6)[:, agent_finger_index[i-1], :].reshape(self.num_envs, -1)
    #             sub_agent_obs.append(torch.cat([sub_dof_pos, sub_dof_vel, sub_dof_force, sub_finger_state, sub_finger_force], dim=1).detach().cpu().numpy())
    #         else:
    #             sub_agent_obs.append(torch.cat([sub_dof_pos, sub_dof_vel, sub_dof_force], dim=1).detach().cpu().numpy())
    #     return sub_agent_obs

    def step(self, actions):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def get_number_of_agents(self):
        return self.num_agents

    def get_env_info(self):
        env_info = {
            'state_shape': self.get_state_size(),
            'obs_shape': self.get_obs_size(),
            'n_actions': self.get_total_actions(),
            'n_agents': self.num_agents,
        }
        return env_info

    @property
    def observation_space(self):
        return self.obs_space

    @property
    def action_space(self):
        return self.act_space

    @property
    def num_envs(self):
        return self.num_environments

    @property
    def num_acts(self):
        return self.num_actions

    @property
    def num_obs(self):
        return self.num_observations


# Python CPU/GPU Class
class ShadowHandMultiVecTaskPython(ShadowHandMultiVecTask):
    def get_state(self):
        return torch.clamp(self.task.states_buf, -self.clip_obs, self.clip_obs).to(self.rl_device)

    def step(self, actions):
        a_hand_actions = actions[0]
        for i in range(1, len(actions)):
            a_hand_actions = torch.hstack((a_hand_actions, actions[i]))

        actions = a_hand_actions
        # actions = torch.flatten(actions, start_dim=1, end_dim=-1)

        actions_tensor = torch.clamp(actions, -self.clip_actions, self.clip_actions)

        self.task.step(actions_tensor)

        hand_obs = []
        obs_buf = torch.clamp(self.task.obs_buf, -self.clip_obs, self.clip_obs).to(self.rl_device)
        # self.process_sub_agent_obs(self.agent_dof_index, self.agent_finger_index, obs_buf)
        hand_obs.append(
            torch.cat([obs_buf[:, : self.num_hand_obs], obs_buf[:, 2 * self.num_hand_obs :]], dim=1)
        )
        hand_obs.append(
            torch.cat(
                [
                    obs_buf[:, self.num_hand_obs : 2 * self.num_hand_obs],
                    obs_buf[:, 2 * self.num_hand_obs :],
                ],
                dim=1,
            )
        )
        state_buf = torch.clamp(self.task.obs_buf, -self.clip_obs, self.clip_obs)
        rewards = self.task.rew_buf.unsqueeze(-1).to(self.rl_device)
        costs = self.task.compute_cost().unsqueeze(-1).to(self.rl_device)
        dones = self.task.reset_buf.to(self.rl_device)

        sub_agent_obs = []
        agent_state = []
        sub_agent_reward = []
        sub_agent_cost = []
        sub_agent_done = []
        sub_agent_info = []
        for i in range(len(self.agent_index[0] + self.agent_index[1])):
            if i < len(self.agent_index[0]):
                sub_agent_obs.append(hand_obs[0])
            else:
                sub_agent_obs.append(hand_obs[1])

            agent_state.append(state_buf)
            sub_agent_reward.append(rewards)
            sub_agent_cost.append(costs)
            sub_agent_done.append(dones)
            sub_agent_info.append(torch.Tensor(0))

        obs_all = torch.transpose(torch.stack(sub_agent_obs), 1, 0)
        state_all = torch.transpose(torch.stack(agent_state), 1, 0)
        reward_all = torch.transpose(torch.stack(sub_agent_reward), 1, 0)
        costs_all = torch.transpose(torch.stack(sub_agent_cost), 1, 0)
        done_all = torch.transpose(torch.stack(sub_agent_done), 1, 0)
        info_all = torch.stack(sub_agent_info)

        return obs_all, state_all, reward_all, costs_all, done_all, info_all, None

    def reset(self):
        actions = 0.01 * (
            1
            - 2
            * torch.rand(
                [self.task.num_envs, self.task.num_actions * 2],
                dtype=torch.float32,
                device=self.rl_device,
            )
        )

        # step the simulator
        self.task.step(actions)

        hand_obs = []
        obs_buf = torch.clamp(self.task.obs_buf, -self.clip_obs, self.clip_obs)
        hand_obs.append(
            torch.cat([obs_buf[:, : self.num_hand_obs], obs_buf[:, 2 * self.num_hand_obs :]], dim=1)
        )
        hand_obs.append(
            torch.cat(
                [
                    obs_buf[:, self.num_hand_obs : 2 * self.num_hand_obs],
                    obs_buf[:, 2 * self.num_hand_obs :],
                ],
                dim=1,
            )
        )
        state_buf = torch.clamp(self.task.obs_buf, -self.clip_obs, self.clip_obs)

        sub_agent_obs = []
        agent_state = []

        for i in range(len(self.agent_index[0] + self.agent_index[1])):
            if i < len(self.agent_index[0]):
                sub_agent_obs.append(hand_obs[0])
            else:
                sub_agent_obs.append(hand_obs[1])
            agent_state.append(state_buf)

        obs = torch.transpose(torch.stack(sub_agent_obs), 1, 0)
        state_all = torch.transpose(torch.stack(agent_state), 1, 0)

        return obs, state_all, None


# VecEnv Wrapper for FreightFranka
class FreightFrankaMultiVecTask:
    def __init__(self, task, rl_device, clip_observations=5.0, clip_actions=1.0):
        self.task = task

        self.num_environments = task.num_envs
        self.num_states = task.num_states
        self.num_actions = task.num_actions
        self.num_freight_obs = task.num_freight_obs
        self.num_franka_obs = task.num_franka_obs

        self.num_freight_observations = task.num_obs - self.num_franka_obs
        self.num_franka_observations = task.num_obs - self.num_freight_obs
        self.nums_share_observations = task.num_obs
        self.num_agents = 2

        self.clip_actions_low = task.franka_dof_lower_limits_tensor
        self.clip_actions_high = task.franka_dof_upper_limits_tensor
        self.rl_device = rl_device
        print('RL device: ', rl_device)

        # COMPATIBILITY
        # self.observation_space = [Box(low=np.array([-10]*self.n_agents), high=np.array([10]*self.n_agents)) for _ in range(self.n_agents)]
        self.obs_space = [
            spaces.Box(low=-np.Inf, high=np.Inf, shape=(self.num_freight_observations,)),
            spaces.Box(low=-np.Inf, high=np.Inf, shape=(self.num_franka_observations,)),
        ]
        self.share_observation_space = [
            spaces.Box(low=-np.Inf, high=np.Inf, shape=(self.nums_share_observations,))
            for _ in range(self.num_agents)
        ]

        self.act_space = tuple(
            [
                spaces.Box(
                    low=np.ones((3,)) * self.clip_actions_low[:3].cpu().numpy(),
                    high=np.ones((3,)) * self.clip_actions_high[:3].cpu().numpy(),
                ),
                spaces.Box(
                    low=np.ones((9,)) * self.clip_actions_low[3:12].cpu().numpy(),
                    high=np.ones((9,)) * self.clip_actions_high[3:12].cpu().numpy(),
                ),
            ]
        )

    def step(self, actions):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def get_number_of_agents(self):
        return self.num_agents

    def get_env_info(self):
        env_info = {
            'state_shape': self.get_state_size(),
            'obs_shape': self.get_obs_size(),
            'n_actions': self.get_total_actions(),
            'n_agents': self.num_agents,
        }
        return env_info

    @property
    def observation_space(self):
        return self.obs_space

    @property
    def action_space(self):
        return self.act_space

    @property
    def num_envs(self):
        return self.num_environments

    @property
    def num_acts(self):
        return self.num_actions

    @property
    def num_obs(self):
        return self.num_observations


# Python CPU/GPU Class
class FreightFrankaMultiVecTaskPython(FreightFrankaMultiVecTask):
    def get_state(self):
        return self.task.states_buf.to(self.rl_device)

    def step(self, actions):
        an_agent_actions = actions[0]
        for i in range(1, len(actions)):
            an_agent_actions = torch.hstack((an_agent_actions, actions[i]))

        actions = an_agent_actions

        actions_tensor = torch.clamp(actions, self.clip_actions_low, self.clip_actions_high)

        obs_buf, rew_buf, cost_buf, reset_buf, _ = self.task.step(actions_tensor)

        sub_agent_obs = []
        # self.process_sub_agent_obs(self.agent_dof_index, self.agent_finger_index, obs_buf)
        num_freight_obs = self.num_freight_obs // 2
        num_franka_obs = self.num_franka_obs // 2
        sub_agent_obs.append(
            torch.cat(
                [
                    obs_buf[:, :num_freight_obs],
                    obs_buf[:, 12 : 12 + num_freight_obs],
                    obs_buf[:, 24:],
                ],
                dim=1,
            )
        )
        sub_agent_obs.append(
            torch.cat(
                [
                    obs_buf[:, num_freight_obs : num_freight_obs + num_franka_obs],
                    obs_buf[:, 12 + num_freight_obs : 12 + num_freight_obs + num_franka_obs],
                    obs_buf[:, 24:],
                ],
                dim=1,
            )
        )
        state_buf = obs_buf
        rewards = rew_buf.unsqueeze(-1).to(self.rl_device)
        costs = cost_buf.unsqueeze(-1).to(self.rl_device)
        dones = reset_buf.to(self.rl_device)

        agent_state = []
        sub_agent_reward = []
        sub_agent_cost = []
        sub_agent_done = []
        sub_agent_info = []
        for i in range(2):
            agent_state.append(state_buf)
            sub_agent_reward.append(rewards)
            sub_agent_cost.append(costs)
            sub_agent_done.append(dones)
            sub_agent_info.append(torch.Tensor(0))

        # obs_all = torch.transpose(torch.stack(sub_agent_obs), 1, 0)
        obs_all = sub_agent_obs
        state_all = torch.transpose(torch.stack(agent_state), 1, 0)
        reward_all = torch.transpose(torch.stack(sub_agent_reward), 1, 0)
        costs_all = torch.transpose(torch.stack(sub_agent_cost), 1, 0)
        done_all = torch.transpose(torch.stack(sub_agent_done), 1, 0)
        info_all = torch.stack(sub_agent_info)

        return obs_all, state_all, reward_all, costs_all, done_all, info_all, None

    def reset(self):
        actions = 0.01 * (
            1
            - 2
            * torch.rand(
                [self.task.num_envs, self.task.num_actions],
                dtype=torch.float32,
                device=self.rl_device,
            )
        )

        # step the simulator
        self.task.step(actions)

        sub_agent_obs = []
        obs_buf = self.task.obs_buf
        # self.process_sub_agent_obs(self.agent_dof_index, self.agent_finger_index, obs_buf)
        num_freight_obs = self.num_freight_obs // 2
        num_franka_obs = self.num_franka_obs // 2
        sub_agent_obs.append(
            torch.cat(
                [
                    obs_buf[:, :num_freight_obs],
                    obs_buf[:, 12 : 12 + num_freight_obs],
                    obs_buf[:, 24:],
                ],
                dim=1,
            )
        )
        sub_agent_obs.append(
            torch.cat(
                [
                    obs_buf[:, num_freight_obs : num_freight_obs + num_franka_obs],
                    obs_buf[:, 12 + num_freight_obs : 12 + num_freight_obs + num_franka_obs],
                    obs_buf[:, 24:],
                ],
                dim=1,
            )
        )
        state_buf = self.task.obs_buf

        agent_state = []
        for i in range(2):
            agent_state.append(state_buf)

        # obs_all = torch.transpose(torch.stack(sub_agent_obs), 1, 0)
        obs_all = sub_agent_obs
        state_all = torch.transpose(torch.stack(agent_state), 1, 0)

        return obs_all, state_all, None
