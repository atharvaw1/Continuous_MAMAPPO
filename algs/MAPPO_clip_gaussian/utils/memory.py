"""Memory buffer script

This manages the memory buffer. 
"""
from copy import deepcopy
   
from .misc import *

@th.jit.script
def compute_gae(b_values: Tensor, value_: Tensor, b_rewards: Tensor, b_dones: Tensor, gamma: float, gae_lambda: float):   
    values_ = th.cat((b_values[1:], value_))
    gamma = gamma * (1 - b_dones)
    deltas = b_rewards + gamma * values_ - b_values
    advantages = th.zeros_like(b_values)
    last_gaelambda = th.zeros_like(b_values[0])
    for t in range(advantages.shape[0] - 1, -1, -1):
        last_gaelambda = advantages[t]  = deltas[t] + gamma[t] * gae_lambda * last_gaelambda
       
    returns = advantages + b_values
 
    return returns, advantages

@th.jit.script
def _discount_cumsum(b_values: Tensor, value_: Tensor, b_rewards: Tensor, b_dones: Tensor, gamma: float, gae_lambda: float):
    returns = th.zeros_like(b_values)
    returns[-1] = value_
    for t in range(returns.shape[0] - 2, -1, -1):
        returns[t] = b_rewards[t] + gamma * (1 - b_dones[t]) * returns[t+1]

    advantages = returns - b_values
    return returns, advantages

class Buffer:
    """
    Class for the Buffer creation
    """

    def __init__(self, env: gym.Env, agents: List[str], size: int, max_steps: int, gamma: float, gae: bool, gae_lambda: float, device: th.device, good_agent: bool = False):
        self.size = size
        self.max_steps = max_steps
        self.o_space, self.a_space = env.observation_space[agents[0]], env.ma_space[agents[0]]

        self.b_obervations = th.zeros((self.size, self.o_space.shape[0])).to(device)

        j_obs_dim = self.o_space.shape[0] * len(agents)
        if good_agent: j_obs_dim = self.o_space.shape[0]
        self.b_j_obervations = th.zeros((self.size, j_obs_dim)).to(device)
        print(self.b_j_obervations.shape)
        self.b_actions = th.zeros((self.size, self.a_space.shape[0])).to(device)
        self.b_means = deepcopy(self.b_actions)
        self.b_stds = deepcopy(self.b_actions)
        self.b_logprobs = th.zeros(self.size, dtype=th.float32).to(device)
        self.b_rewards = deepcopy(self.b_logprobs)
        self.b_values = deepcopy(self.b_logprobs)
        self.b_dones = deepcopy(self.b_logprobs)
        self.idx = 0

        self.gamma = gamma
        self.gae = gae
        self.gae_lambda = gae_lambda

        self.device = device

    def store(self, observation, j_observation, action, mean, std, logprob, reward, value, done):
        self.b_obervations[self.idx] = observation  
        self.b_j_obervations[self.idx] = j_observation      
        self.b_actions[self.idx] = action
        self.b_means[self.idx] = mean
        self.b_stds[self.idx] = std
        self.b_logprobs[self.idx] = logprob
        self.b_rewards[self.idx] = reward
        self.b_values[self.idx] = value
        self.b_dones[self.idx] = done
        self.idx += 1

    def compute_mc(self, value_):
        mc = compute_gae if self.gae else _discount_cumsum
        (
            self.returns, 
            self.advantages 
        ) = mc(self.b_values, value_, self.b_rewards, self.b_dones, self.gamma, self.gae_lambda)
        
    def sample(self):
        n_episodes = int(self.size / self.max_steps)
        
        return {
            'observations': self.b_obervations.reshape((n_episodes, self.max_steps, -1)),     # TODO reshape as needed
            'j_observations': self.b_j_obervations.reshape((n_episodes, self.max_steps, -1)),
            'actions': self.b_actions.reshape((n_episodes, self.max_steps, -1)),
            'means': self.b_means.reshape((n_episodes, self.max_steps, -1)),
            'stds': self.b_stds.reshape((n_episodes, self.max_steps, -1)),
            'logprobs': self.b_logprobs.reshape((n_episodes, self.max_steps)), 
            'values': self.b_values.reshape((n_episodes, self.max_steps)),
            'returns': self.returns.reshape((n_episodes, self.max_steps)),
            'advantages': self.advantages.reshape((n_episodes, self.max_steps)),
        }

    def clear(self):
        self.idx = 0

       
