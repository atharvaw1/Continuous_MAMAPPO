import time
from abc import ABC, abstractmethod
from typing import Dict, List

import numpy as np
from gym.spaces import Box

Array = np.array


def _array_from_dict(data):
    return np.array(list(data.values())).squeeze()


def make_env(scenario_name, **kwargs):
    '''
    Creates a MultiAgentEnv object as env. This can be used similar to a gym
    environment by calling env.reset() and env.step().
    Use env.render() to view the environment on the screen.
    Input:
        scenario_name   :   name of the scenario from ./scenarios/ to be Returns
                            (without the .py extension)
        kwargs       :    {num_good : int, num_adversaries : int, num_obstacles : int, max_cycles : int, continuous_actions : bool}
    Some useful env properties (see environment.py):
        .observation_space  :   Returns the observation space for each agent
        .action_space       :   Returns the action space for each agent
        .n                  :   Returns the number of Agents
    '''
    from envs.mpe.environment import MultiAgentEnv
    import envs.mpe.scenarios as scenarios

    # load scenario from script
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    # create world
    world = scenario.make_world(**kwargs)
    # create multiagent environment
    env = MultiAgentEnv(world, kwargs['max_steps'], scenario.reset_world, scenario.reward, scenario.observation,
                        scenario.collision, discrete_action_space=True)

    return env


class MaWrapper(ABC):
    """Macro action wrapper for pettingzoo's env."""

    def __init__(self, env_id: str, seed: int = 0, **kwargs) -> None:
        """
        Args:
            <arg name> (<arg type>): <description>
            print_cols (bool): A flag used to print the columns to the console
                (default is False)
        """

        self.env = make_env(env_id, **kwargs)
        self.num_envs = 1

        self.seed(seed)
        self.reset()

        self.agent_ids = [agent.name for agent in self.env.agents]

        self.steps, self.max_steps = 0, kwargs['max_steps']
        self.ma_threshold = 0.05  # threshold for waypoint reaching
        self.agent_size = 0.1
        self.ma_step = np.zeros(self.env.n)
        self.ma_done = np.ones(self.env.n, dtype=bool)

    def seed(self, seed: int = 0):
        self.env.seed(seed)

    def reset(self):
        self.state = self.env.reset()
        return self.state

    @abstractmethod
    def step(self, actions: Dict[str, List[float]]):
        raise NotImplementedError

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()

    def _get_positions(self):
        """Get agents' positions.

        Returns:
            positions (Array[List[float]]): agents positions
        """
        return np.array(self.state)[:, 2:4]

    def _get_collisions(self):
        """Check predators collisions.

        Returns:
            collisions (Array[int]): indicator cost signal for collisions
        """
        cost = np.zeros(self.env.n)
        agent_pos = self._get_positions()

        # Count Collisions
        for idx_0 in range(self.env.n):
            for idx_1 in range(self.env.n):
                if idx_0 == idx_1: continue
                dist = np.linalg.norm(agent_pos[idx_0] - agent_pos[idx_1])

                if dist < self.agent_size:
                    cost[idx_0] += 1
        return cost

    @property
    def observation_space(self):
        """Get the agents' observation spaces.

        Returns:
            dict: a dictionary with agents' observation spaces
        """
        return {k: obs_space for k, obs_space in zip(self.agent_ids, self.env.observation_space)}

    @property
    def action_space(self):
        """Get the agents' action spaces.

        Returns:
            dict: a dictionary with agents' action spaces
        """
        return {k: act_space for k, act_space in zip(self.agent_ids, self.env.action_space)}

    @property
    def agents(self):
        """Get the agents' id.

        Returns:
            list: a list with agents' id
        """
        return self.env.agents

    @property
    def n(self):
        """Get the n° agents'

        Returns:
            int: the number of agents
        """
        return self.env.n


class MaSpreadWrapper(MaWrapper):
    """Macro action wrapper for pettingzoo's Simple Spread env.
    
    Observations [self_vel (2), self_pos (2), landmark_rel_positions (2*3), other_agent_rel_positions (2*2)]
    Original discrete actions [no_action, move_right, move_left, move_up, move_down]

    Environment coordinates are (0, 0) in the center, x increases going right, y increases going up

    Attributes: -> all the self.
        likes_spam: A boolean indicating if we like SPAM or not.
        eggs: An integer count of the eggs we have laid.
    """

    def __init__(self, seed: int = 0, max_steps: int = 100) -> None:
        """
        Args:
            <arg name> (<arg type>): <description>
            print_cols (bool): A flag used to print the columns to the console
                (default is False)
        """
        self.env_params = {'num_agents': 3, 'num_landmarks': 3, 'max_steps': max_steps}
        super().__init__('simple_spread', seed, **self.env_params)

        ma_shape = (2,)  # Each agent has a 2-dim macro action with (x, y) coordinates
        self.ma_space = {k: Box(low=-1, high=1, shape=ma_shape, dtype=np.float32) for k in self.agent_ids}
        self.tg_pos = np.zeros((self.env.n, ma_shape[0]))

    def step(self, actions: Dict[str, List[float]]):
        """Perform actions in the environment.

        Cast the continuous macro action to discrete commands to reach the target location and perform the step.

        Returns:
            observations (Dict[str, List[float]]): agents' observations
            rewards (Dict[str, List[float]]): agents' rewards
            done (bool): whether episode is done
            info (...): a dictionary with agents' observations
        """

        self.steps += 1
        self.ma_step[self.ma_done] = 0  # reset ma step counter if ma ended in the last step
        tg_pos = _array_from_dict(actions)
        current_pos = self._get_positions()
        delta_pos = tg_pos - current_pos

        # print(f"Delta pos: {delta_pos}")
        x_actions = np.where(delta_pos[:, 0] < 0, 1, 2)
        x_actions = np.where(np.abs(delta_pos[:, 0]) < self.ma_threshold, 0, x_actions)
        y_actions = np.where(delta_pos[:, 1] < 0, 3, 4)
        y_actions = np.where(np.abs(delta_pos[:, 1]) < self.ma_threshold, 0, y_actions)

        # # Non-zero primitive actions policy
        # actions_nonnull = np.where(x_actions > 0, x_actions, y_actions)
        # rand = np.random.rand(x_actions.shape[0])
        # actions_nonnull = np.where((x_actions > 0) & (y_actions > 0) & (rand < 0.5), y_actions, actions_nonnull)
        # self.state, reward, done, info = self.env.step(actions_nonnull)


        # print(f"Chosen waypoint: {actions}")
        # A ma ends when the {xy}_actions are 0
        actions = np.sum(np.stack((x_actions, y_actions), axis=-1), axis=-1)

        # print(f"Sum discrete action: {actions}")
        # print(f"Actions without 0s: {actions_nonnull}")
        # print(f"Action x: {x_actions}")
        # print(f"Action y: {y_actions}")

        self.ma_done = actions == 0
        self.ma_step += 1

        # If an agent doesn't have to move on x, it should move on y because otherwise it biases the reward
        print(x_actions)
        _, reward_x, _, info_x = self.env.step(x_actions)
        # _, reward_x, _, info_x = self.env.step(x_actions)
        # print(f"Step size x: {s[0][2:4] - current_pos[0]}")
        # self.env.render()
        # print(f"Macro action done: {self.ma_done}")
        # input()
        reward_x /= self.env.n  # Because we have joint rew and the env is still summing agents' rewards
        # self.env.render()

        self.state, reward_y, done, info = self.env.step(y_actions)
        # time.sleep(0.2)
        # print(self.state[0][2:4])
        # print(s[0][2:4])
        # print(f"Step size y: {self.state[0][2:4] - s[0][2:4]}")
        reward_y /= self.env.n
        # self.env.render()
        # time.sleep(0.2)

        reward = reward_x + reward_y

        info['ma_step'] = self.ma_step
        info['ma_done'] = self.ma_done
        info['cost'] += info_x['cost']

        return self.state, reward, done, info


class MaReferenceWrapper(MaWrapper):
    """Macro action wrapper for pettingzoo's Simple Reference env.

    Observations [self_vel (2), self_pos (2), all_landmark_rel_positions (2*3), goal_id (1), communication (10)]
    (21,):
        X goal is the rgb color of the landmark (i.e., useless, would be better to have its code or a one hot)
        - goal is now the landmark id
        - communication is a one hot of 10 targets (even though we have only 3 landmarks)

    Original discrete actions [[no_action, move_left, move_right, move_down, move_up], [say_0, say_1, say_2, say_3, say_4, say_5, say_6, say_7, say_8, say_9]]

    Environment coordinates are (0, 0) in the center, x increases going right, y increases going up

    # Cost: communication constraint triggers a cost when agents communicate a landmark > 3 (i.e., that does not exist)
    # Cost: collision constraint triggers a cost when agents collide

    Attributes: -> all the self.
        likes_spam: A boolean indicating if we like SPAM or not.
        eggs: An integer count of the eggs we have laid.
    """

    def __init__(self, seed: int = 0, max_steps: int = 0) -> None:
        """
        Args:
            <arg name> (<arg type>): <description>
            print_cols (bool): A flag used to print the columns to the console
                (default is False)
        """
        self.env_params = {'max_steps': max_steps}
        super().__init__('simple_reference', seed, **self.env_params)

        ma_shape = (3,)
        # ma_shape = [2, 1]  # Each agent has a 2-dim macro action with (x, y) coordinates
        self.ma_space = {k: Box(low=np.array([-1, -1, 0]), high=1, shape=(np.sum(ma_shape),), dtype=np.float32) for k in
                         self.agent_ids}
        self.tg_pos = np.zeros((self.env.n, 2))
        self.tg_goal = np.zeros((self.env.n, 1))

    def step(self, actions: Dict[str, List[float]]):
        """Perform actions in the environment.

        Cast the continuous macro action to discrete commands to reach the target location and perform the step.

        Returns:
            observations (Dict[str, List[float]]): agents' observations
            rewards (Dict[str, List[float]]): agents' rewards
            done (bool): whether episode is done
            info (...): a dictionary with agents' observations
        """

        self.steps += 1
        self.ma_step[self.ma_done] = 0  # reset ma step counter if ma ended in the last step

        actions = _array_from_dict(actions)

        tg_pos = actions[:, :2]
        tg_goal = actions[:, -1]

        comm_actions = self._get_comm(tg_goal)

        current_pos = self._get_positions()
        delta_pos = tg_pos - current_pos
        x_actions = np.where(delta_pos[:, 0] > 0, 1, 2)
        x_actions = np.where(np.abs(delta_pos[:, 0]) < self.ma_threshold, 0, x_actions)
        y_actions = np.where(delta_pos[:, 1] > 0, 3, 4)
        y_actions = np.where(np.abs(delta_pos[:, 1]) < self.ma_threshold, 0, y_actions)

        # A ma ends when the {xy}_actions are 0
        actions = np.sum(np.stack((x_actions, y_actions), axis=-1), axis=-1)
        self.ma_done = actions == 0
        self.ma_step += 1

        # Fix the action shape to be a MultiDiscrete

        x_actions = np.stack((x_actions, comm_actions), axis=-1)
        y_actions = np.stack((y_actions, comm_actions), axis=-1)

        _, reward_x, _, info_x = self.env.step(x_actions)
        reward_x /= self.env.n
        self.state, reward_y, done, info = self.env.step(y_actions)
        reward_y /= self.env.n

        reward = reward_x + reward_y
        info['ma_step'] = self.ma_step
        info['ma_done'] = self.ma_done
        info['cost'] += info_x['cost']

        if self.steps >= self.max_steps:
            done = np.ones(self.env.n)
            self.steps = 0

        return self.state, reward, done, info

    def _get_comm(self, comm_actions):
        """Get integer communication action from [0, 1] continuous action.

        Returns:
            landmark_id (Array[int]): target landmark id to communicate
        """

        return (comm_actions.clip(0, 1 - 1e-5) * 10).astype(np.int32)


# class MaSpeakerWrapper(MaWrapper):
#     """Macro action wrapper for pettingzoo's Simple Speaker env.
#
#     Observations
#         - (11,) speaker (agent 0) [goal_id (1) 0-pad (10)]
#         - (11,) listener (agent 1) [self_vel (2), self_pos (2), all_landmark_rel_positions (2*3), communication (1)]
#
#     Original discrete actions [say_0, say_1, say_2], [no_action, move_left, move_right, move_down, move_up]
#
#     Environment coordinates are (0, 0) in the center, x increases going right, y increases going up
#
#     # Cost: ? Possibly when communicating the right landmark with end prob > 0.3, atm just if end prob > 0.3
#
#     Attributes: -> all the self.
#         likes_spam: A boolean indicating if we like SPAM or not.
#         eggs: An integer count of the eggs we have laid.
#     """
#
#     def __init__(self, seed: int = 0, max_steps: int = 100) -> None:
#         """
#         Args:
#             <arg name> (<arg type>): <description>
#             print_cols (bool): A flag used to print the columns to the console
#                 (default is False)
#         """
#         super().__init__('simple_speaker_listener', seed, max_steps)
#
#         ma_shape = (2,)
#         self.ma_space = {
#             'agent 0': Box(low=0, high=1, shape=ma_shape, dtype=np.float32),
#             'agent 1': Box(low=-1, high=1, shape=ma_shape, dtype=np.float32),
#         }
#         self.tg_pos = np.zeros((self.env.n, ma_shape[0]))
#         self.tg_goal = np.zeros((self.env.n, ma_shape[0]))
#
#         self.agent_size = 0.15
#
#     def step(self, actions: Dict[str, List[float]]):
#         """Perform actions in the environment.
#
#         ...
#
#         Returns:
#             observations (Dict[str, List[float]]): agents' observations
#             rewards (Dict[str, List[float]]): agents' rewards
#             done (bool): whether episode is done
#             info (...): a dictionary with agents' observations
#             ...
#         """
#
#         self.steps += 1
#         self.ma_step[self.ma_done] = 0  # reset ma step counter if ma ended in the last step
#
#         actions = _array_from_dict(actions)
#
#         # Speaker
#         tg_comm, comm_prob = actions[0]
#         comm_action = self._get_comm(tg_comm)
#         self.ma_done[0] = np.random.rand() < comm_prob  # Speaker ma ends probabilistically based on the 2° ma
#
#         # Listener
#         tg_pos = actions[1]
#         current_pos = self._get_positions()[1]  # Get the listener position
#
#         delta_pos = tg_pos - current_pos
#         x_actions = np.where(delta_pos[0] > 0, 1, 2)
#         x_actions = np.where(np.abs(delta_pos[0]) < self.ma_threshold, 0, x_actions)
#         y_actions = np.where(delta_pos[1] > 0, 3, 4)
#         y_actions = np.where(np.abs(delta_pos[1]) < self.ma_threshold, 0, y_actions)
#
#         # Listener ma ends when the {xy}_actions are 0, i.e., waypont reached
#         actions = np.sum(np.stack((x_actions, y_actions), axis=-1), axis=-1)
#         self.ma_done[1] = actions == 0
#         self.ma_step += 1
#
#         # Concat the agents' actions
#         x_actions = np.stack((comm_action, x_actions), axis=-1)
#         y_actions = np.stack((comm_action, y_actions), axis=-1)
#
#         _, reward_x, _, _ = self.env.step(x_actions)
#         reward_x /= self.env.n  # Because we have joint rew and the env is still summing agents' rewards
#         self.state, reward_y, done, info = self.env.step(y_actions)
#         reward_y /= self.env.n  # Because we have joint rew and the env is still summing agents' rewards
#
#         reward = reward_x + reward_y
#         cost = [0, int(comm_prob > 0.3)]
#         info['ma_step'] = self.ma_step
#         info['ma_done'] = self.ma_done
#         info['cost'] = cost
#
#         if self.steps >= self.max_steps:
#             done = np.ones(self.env.n)
#             self.steps = 0
#
#         return self.state, reward, done, info
#
#     def _get_comm(self, comm_actions):
#         """Get integer communication action from [0, 1] continuous action.
#
#         Returns:
#             landmark_id (Array[int]): target landmark id [0, 0.5, 1.0] to communicate
#         """
#
#         return (comm_actions.clip(0, .99) // 0.33).astype(np.int32)
#
#
# class MaTagWrapper(MaWrapper):
#     """Macro action wrapper for pettingzoo's Simple Tag (Predator-Prey) env.
#
#     Observations [self_vel (2), self_pos (2), landmark_rel_positions (2*n_landmarks), other_agent_rel_positions (2*n_agents), other_agent_velocities (2*n° preys)]
#     (16,) for adversaries and (14,) for agent (adversaries observe the velocity of the prey)
#     Original discrete actions [no_action, move_right, move_left, move_up, move_down]
#
#     Environment coordinates are (0, 0) in the center, x increases going right, y increases going up
#
#     Attributes: -> all the self.
#         likes_spam: A boolean indicating if we like SPAM or not.
#         eggs: An integer count of the eggs we have laid.
#     """
#
#     def __init__(self, seed: int = 0, max_steps: int = 100) -> None:
#         """
#         Args:
#             <arg name> (<arg type>): <description>
#             print_cols (bool): A flag used to print the columns to the console
#                 (default is False)
#         """
#         self.env_params = {
#             'num_good': 1,
#             'num_adversaries': 3,
#             'num_obstacles': 2,
#         }
#         super().__init__('simple_tag', seed, max_steps, **self.env_params)
#
#         ma_shape = (2,)  # Each agent has a 2-dim macro action with (x, y) coordinates
#         self.ma_space = {k: Box(low=-1, high=1, shape=ma_shape, dtype=np.float32) for k in self.agent_ids}
#         self.tg_pos = np.zeros((self.env.n, ma_shape[0]))
#
#         self.pred_size = 0.15
#
#     def step(self, actions: Dict[str, List[float]]):
#         """Perform actions in the environment.
#
#         Cast the continuous macro action to discrete commands to reach the target location and perform the step.
#
#         Returns:
#             observations (Dict[str, List[float]]): agents' observations
#             rewards (Dict[str, List[float]]): agents' rewards
#             done (bool): whether episode is done
#             info (...): a dictionary with agents' observations
#         """
#
#         self.steps += 1
#         self.ma_step[self.ma_done] = 0  # reset ma step counter if ma ended in the last step
#
#         tg_pos = _array_from_dict(actions)
#         current_pos = self._get_positions()
#
#         delta_pos = tg_pos - current_pos
#         x_actions = np.where(delta_pos[:, 0] > 0, 1, 2)
#         x_actions = np.where(np.abs(delta_pos[:, 0]) < self.ma_threshold, 0, x_actions)
#         y_actions = np.where(delta_pos[:, 1] > 0, 3, 4)
#         y_actions = np.where(np.abs(delta_pos[:, 1]) < self.ma_threshold, 0, y_actions)
#
#         # A ma ends when the {xy}_actions are 0
#         actions = np.sum(np.stack((x_actions, y_actions), axis=-1), axis=-1)
#         self.ma_done = actions == 0
#         self.ma_step += 1
#
#         _, reward_x, _, _ = self.env.step(x_actions)
#         cost_x = self._get_adv_collisions()
#         self.state, reward_y, done, info = self.env.step(y_actions)
#         cost_y = self._get_adv_collisions()
#         # self.env.render()
#         # time.sleep(0.1)
#
#         reward = reward_x + reward_y
#         info['ma_step'] = self.ma_step
#         info['ma_done'] = self.ma_done
#         info['cost'] = cost_x + cost_y
#
#         if self.steps >= self.max_steps:
#             done = np.ones(self.env.n)
#             self.steps = 0
#
#         return self.state, reward, done, info
#
#     def _get_adv_collisions(self):
#         """Check predators collisions.
#
#         Returns:
#             collisions (Array[int]): indicator cost signal for collisions
#         """
#         cost = np.zeros(self.env_params['num_adversaries'])
#         pred_pos = self._get_positions()[:-1]
#
#         # Count Collisions
#         for idx_0 in range(self.env_params['num_adversaries']):
#             for idx_1 in range(self.env_params['num_adversaries']):
#                 if idx_0 == idx_1: continue
#                 dist = np.linalg.norm(pred_pos[idx_0] - pred_pos[idx_1])
#
#                 if dist < self.pred_size:
#                     cost[idx_0] += 1
#         return cost
#
#
# class MaPushWrapper(MaWrapper):
#     """Macro action wrapper for pettingzoo's Simple Push env.
#
#     Agent observation space: [self_vel, goal_rel_position, goal_landmark_id, all_landmark_rel_positions,
#     landmark_ids, other_agent_rel_positions]
#     Adversary observation space: [self_vel, all_landmark_rel_positions, other_agent_rel_positions]
#     Original Agent primitve action space: [no_action, move_left, move_right, move_down, move_up]
#     Original Adversary primitve action space: [no_action, move_left, move_right, move_down, move_up]
#
#     Environment coordinates are (0, 0) in the center, x increases going right, y increases going up
#
#     """
#
#     def __init__(self, seed: int = 0, max_steps: int = 100) -> None:
#         """
#         Args:
#             <arg name> (<arg type>): <description>
#             print_cols (bool): A flag used to print the columns to the console
#                 (default is False)
#         """
#         self.env_params = {
#             'num_good': 1,
#             'num_adversaries': 2,
#             'num_landmarks': 1
#         }
#         super().__init__('simple_push', seed, max_steps, **self.env_params)
#
#         ma_shape = (2,)  # Each agent has a 2-dim macro action with (x, y) coordinates
#         self.ma_space = {k: Box(low=-1, high=1, shape=ma_shape, dtype=np.float32) for k in self.agent_ids}
#         self.tg_pos = np.zeros((self.env.n, ma_shape[0]))
#         self.pred_size = self.agent_size
#
#     def step(self, actions: Dict[str, List[float]]):
#         """Perform actions in the environment.
#
#         Cast the continuous macro action to discrete commands to reach the target location and perform the step.
#
#         Returns:
#             observations (Dict[str, List[float]]): agents' observations
#             rewards (Dict[str, List[float]]): agents' rewards
#             done (bool): whether episode is done
#             info (...): a dictionary with agents' observations
#         """
#         self.steps += 1
#         self.ma_step[self.ma_done] = 0  # reset ma step counter if ma ended in the last step
#
#         tg_pos = _array_from_dict(actions)
#         current_pos = self._get_positions()
#
#         delta_pos = tg_pos - current_pos
#         x_actions = np.where(delta_pos[:, 0] > 0, 1, 2)
#         x_actions = np.where(np.abs(delta_pos[:, 0]) < self.ma_threshold, 0, x_actions)
#         y_actions = np.where(delta_pos[:, 1] > 0, 3, 4)
#         y_actions = np.where(np.abs(delta_pos[:, 1]) < self.ma_threshold, 0, y_actions)
#
#         # A ma ends when the {xy}_actions are 0
#         actions = np.sum(np.stack((x_actions, y_actions), axis=-1), axis=-1)
#         self.ma_done = actions == 0
#         self.ma_step += 1
#
#         _, reward_x, _, _ = self.env.step(x_actions)
#         cost_x = self._get_agent_collisions()
#         cost_x_adv = self._get_adv_collisions()
#         self.state, reward_y, done, info = self.env.step(y_actions)
#         cost_y = self._get_agent_collisions()
#         cost_y_adv = self._get_adv_collisions()
#         # self.env.render()
#         # time.sleep(0.1)
#         reward = reward_x + reward_y
#         info['ma_step'] = self.ma_step
#         info['ma_done'] = self.ma_done
#         info['cost'] = cost_x + cost_y
#         info['cost_adv'] = cost_x_adv + cost_y_adv
#
#         if self.steps >= self.max_steps:
#             done = np.ones(self.env.n)
#             self.steps = 0
#
#         return self.state, reward, done, info
#
#     def _get_agent_collisions(self):
#         """Check predators collisions.
#
#         Returns:
#             collisions (Array[int]): indicator cost signal for collisions
#         """
#         cost = np.zeros(self.env_params['num_good'])
#         agent_pos = self._get_positions()[self.env_params['num_adversaries'] - 1:]
#         # Count Collisions
#         for idx_0 in range(self.env_params['num_good']):
#             for idx_1 in range(self.env_params['num_good']):
#                 if idx_0 == idx_1: continue
#                 dist = np.linalg.norm(agent_pos[idx_0] - agent_pos[idx_1])
#                 if dist < self.agent_size:
#                     cost[idx_0] += 1
#         return cost
#
#     def _get_adv_collisions(self):
#         """Check predators collisions.
#
#         Returns:
#             collisions (Array[int]): indicator cost signal for collisions
#         """
#         cost = np.zeros(self.env_params['num_adversaries'])
#         pred_pos = self._get_positions()[:self.env_params['num_adversaries']]
#         # Count Collisions
#         for idx_0 in range(self.env_params['num_adversaries']):
#             for idx_1 in range(self.env_params['num_adversaries']):
#                 if idx_0 == idx_1: continue
#                 dist = np.linalg.norm(pred_pos[idx_0] - pred_pos[idx_1])
#
#                 if dist < self.pred_size:
#                     cost[idx_0] += 1
#         return cost
#
#
# class MaAdversaryWrapper(MaWrapper):
#     """Macro action wrapper for pettingzoo's Simple Adversary env.
#
#     Agent observation space: [self_pos, self_vel, goal_rel_position, landmark_rel_position, other_agent_rel_positions]
#     Adversary observation space: [landmark_rel_position, other_agents_rel_positions]
#     Agent action space: [no_action, move_left, move_right, move_down, move_up]
#     Adversary action space: [no_action, move_left, move_right, move_down, move_up]
#
#     Environment coordinates are (0, 0) in the center, x increases going right, y increases going up
#
#     """
#
#     def __init__(self, seed: int = 0, max_steps: int = 100) -> None:
#         """
#         Args:
#             <arg name> (<arg type>): <description>
#             print_cols (bool): A flag used to print the columns to the console
#                 (default is False)
#         """
#         self.env_params = {
#             'num_good': 2,
#             'num_adversaries': 1,
#         }
#         super().__init__('simple_adversary', seed, max_steps, **self.env_params)
#
#         ma_shape = (2,)  # Each agent has a 2-dim macro action with (x, y) coordinates
#         self.ma_space = {k: Box(low=-1, high=1, shape=ma_shape, dtype=np.float32) for k in self.agent_ids}
#         self.tg_pos = np.zeros((self.env.n, ma_shape[0]))
#         self.pred_size = self.agent_size
#
#     def step(self, actions: Dict[str, List[float]]):
#         """Perform actions in the environment.
#
#         Cast the continuous macro action to discrete commands to reach the target location and perform the step.
#
#         Returns:
#             observations (Dict[str, List[float]]): agents' observations
#             rewards (Dict[str, List[float]]): agents' rewards
#             done (bool): whether episode is done
#             info (...): a dictionary with agents' observations
#         """
#         self.steps += 1
#         self.ma_step[self.ma_done] = 0  # reset ma step counter if ma ended in the last step
#
#         tg_pos = _array_from_dict(actions)
#         current_pos = self._get_positions()
#
#         delta_pos = tg_pos - current_pos
#
#         x_actions = np.where(delta_pos[:, 0] > 0, 1, 2)
#         x_actions = np.where(np.abs(delta_pos[:, 0]) < self.ma_threshold, 0, x_actions)
#         y_actions = np.where(delta_pos[:, 1] > 0, 3, 4)
#         y_actions = np.where(np.abs(delta_pos[:, 1]) < self.ma_threshold, 0, y_actions)
#
#         # A ma ends when the {xy}_actions are 0
#         actions = np.sum(np.stack((x_actions, y_actions), axis=-1), axis=-1)
#         self.ma_done = actions == 0
#         self.ma_step += 1
#
#         _, reward_x, _, _ = self.env.step(x_actions)
#         cost_x = self._get_agent_collisions()
#         cost_x_adv = self._get_adv_collisions()
#         self.state, reward_y, done, info = self.env.step(y_actions)
#         cost_y = self._get_agent_collisions()
#         cost_y_adv = self._get_adv_collisions()
#         # self.env.render()
#         # time.sleep(0.1)
#
#         reward = reward_x + reward_y
#         info['ma_step'] = self.ma_step
#         info['ma_done'] = self.ma_done
#         info['cost'] = cost_x + cost_y
#         info['cost_adv'] = cost_x_adv + cost_y_adv
#
#         if self.steps >= self.max_steps:
#             done = np.ones(self.env.n)
#             self.steps = 0
#
#         return self.state, reward, done, info
#
#     def _get_agent_collisions(self):
#         """Check predators collisions.
#
#         Returns:
#             collisions (Array[int]): indicator cost signal for collisions
#         """
#         cost = np.zeros(self.env_params['num_good'])
#         agent_pos = self._get_positions()[self.env_params['num_adversaries'] - 1:]
#         # Count Collisions
#         for idx_0 in range(self.env_params['num_good']):
#             for idx_1 in range(self.env_params['num_good']):
#                 if idx_0 == idx_1: continue
#                 dist = np.linalg.norm(agent_pos[idx_0] - agent_pos[idx_1])
#                 if dist < self.agent_size:
#                     cost[idx_0] += 1
#         return cost
#
#     def _get_adv_collisions(self):
#         """Check predators collisions.
#
#         Returns:
#             collisions (Array[int]): indicator cost signal for collisions
#         """
#         cost = np.zeros(self.env_params['num_adversaries'])
#         pred_pos = self._get_positions()[:self.env_params['num_adversaries']]
#         # Count Collisions
#         for idx_0 in range(self.env_params['num_adversaries']):
#             for idx_1 in range(self.env_params['num_adversaries']):
#                 if idx_0 == idx_1: continue
#                 dist = np.linalg.norm(pred_pos[idx_0] - pred_pos[idx_1])
#
#                 if dist < self.pred_size:
#                     cost[idx_0] += 1
#         return cost