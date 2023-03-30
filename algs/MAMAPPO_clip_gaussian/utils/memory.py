"""Memory buffer script

This manages the memory buffer. 
"""
from copy import deepcopy

from torch.nn.utils.rnn import pad_sequence
from .misc import *

def _get_pad_and_mask_from_obs(observations):
    zeropad_observations = pad_sequence(observations, padding_value=th.tensor(float('0')), batch_first=True)
    nanpad_observations = pad_sequence(observations, padding_value=th.tensor(float('nan')), batch_first=True)
   
    return zeropad_observations, ~th.isnan(nanpad_observations).any(-1)

def compute_gae(idx: int, b_values: Tensor, value_: Tensor, b_rewards: Tensor, b_dones: Tensor, gamma: float, gae_lambda: float): 

    values_ = th.cat((b_values[1:idx], value_))
   
    gamma = gamma * (1 - b_dones[:idx])     # gamma is wrong due to the temporal abstraction?
    deltas = b_rewards[:idx] + gamma * values_ - b_values[:idx]

    advantages = th.zeros_like(b_values[:idx])
    last_gaelambda = th.zeros_like(b_values[0])
    
    for t in range(advantages.shape[0] - 1, -1, -1):
        last_gaelambda = advantages[t]  = deltas[t] + gamma[t] * gae_lambda * last_gaelambda
    
    returns = advantages + b_values[:idx]
    
    return returns, advantages

#@th.jit.script
def _discount_cumsum(idx: int, b_values: Tensor, value_: Tensor, b_rewards: Tensor, b_dones: Tensor, gamma: float, gae_lambda: float):
    returns = th.zeros_like(b_values[:idx])
   
    returns[-1] = value_
   
    for t in range(returns.shape[0] - 2, -1, -1):
        returns[t] = b_rewards[:idx][t] + gamma * (1 - b_dones[:idx][t]) * returns[t+1]

    advantages = returns - b_values[:idx]
    return returns, advantages

class Buffer:
    """
    Class for the Buffer creation
    """

    def __init__(self, env: gym.Env, agents: List[str], size: int, max_steps: int, gamma: float, gae: bool, gae_lambda: float, device: th.device, good_agent: bool = False):
        self.size = size
        self.max_steps = max_steps
        self.o_space, self.a_space = env.observation_space[agents[0]], env.ma_space[agents[0]]

        self.b_observations = th.zeros((self.size, self.o_space.shape[0] + 2)).to(device)

        j_obs_dim = (self.o_space.shape[0] + 2) * len(agents)
        if good_agent: j_obs_dim = self.o_space.shape[0] + 2
        self.b_j_obervations = th.zeros((self.size, j_obs_dim)).to(device)
        self.b_actions = th.zeros((self.size, self.a_space.shape[0])).to(device)
        self.b_means = deepcopy(self.b_actions)
        self.b_stds = deepcopy(self.b_actions)
        self.b_logprobs = th.zeros(self.size, dtype=th.float32).to(device)
        self.b_rewards = deepcopy(self.b_logprobs)
        self.b_values = deepcopy(self.b_logprobs)
        self.b_dones = deepcopy(self.b_logprobs)


        self.idx = 0
        self.ep_lens = []

        self.gamma = gamma
        self.gae = gae
        self.gae_lambda = gae_lambda

        self.device = device

    def store(self, observation, j_observation, action, mean, std, logprob, reward, value, done):
        self.b_observations[self.idx] = observation  
        self.b_j_obervations[self.idx] = j_observation      
        self.b_actions[self.idx] = action
        self.b_means[self.idx] = mean
        self.b_stds[self.idx] = std
        self.b_logprobs[self.idx] = logprob
        self.b_rewards[self.idx] = reward
        self.b_values[self.idx] = value
        self.b_dones[self.idx] = done
        self.idx += 1

    def _store_ep_len(self):
        self.ep_lens.append(self.idx)

    def compute_mc(self, value_):
      
        mc = compute_gae if self.gae else _discount_cumsum
        (
            self.returns, 
            self.advantages 
        ) = mc(self.idx, self.b_values, value_, self.b_rewards, self.b_dones, self.gamma, self.gae_lambda)

    '''    
    def sample(self):
        n_episodes = int(self.size / self.max_steps)
        
        return {
            'observations': self.b_observations.reshape((n_episodes, self.max_steps, -1)),     # TODO reshape as needed
            'j_observations': self.b_j_obervations.reshape((n_episodes, self.max_steps, -1)),
            'actions': self.b_actions.reshape((n_episodes, self.max_steps, -1)),
            'means': self.b_means.reshape((n_episodes, self.max_steps, -1)),
            'stds': self.b_stds.reshape((n_episodes, self.max_steps, -1)),
            'logprobs': self.b_logprobs.reshape((n_episodes, self.max_steps)), 
            'values': self.b_values.reshape((n_episodes, self.max_steps)),
            'returns': self.returns.reshape((n_episodes, self.max_steps)),
            'advantages': self.advantages.reshape((n_episodes, self.max_steps)),
        }
    '''

    def sample(self):        
        # Get the len of each episode
        
        # TODO check why some ep_lens have the last two elements that are equal
        #if self.ep_lens[-1] == self.ep_lens[-2]:
        #    self.ep_lens = self.ep_lens[:-1]
        self.ep_lens[1:] = np.array(self.ep_lens)[1:] - np.array(self.ep_lens)[:-1]

        b_observations = th.split(self.b_observations[:self.idx], self.ep_lens)
        b_j_obervations = th.split(self.b_j_obervations[:self.idx], self.ep_lens)
        b_actions = th.split(self.b_actions[:self.idx], self.ep_lens)
        b_logprobs = self.b_logprobs[:self.idx]
        advantages = self.advantages[:self.idx]
        returns = self.returns[:self.idx]

        pad_observations, mask_observations  = _get_pad_and_mask_from_obs(b_observations)
        pad_j_observations, mask_j_observations  = _get_pad_and_mask_from_obs(b_j_obervations)
        pad_actions, mask_actions  = _get_pad_and_mask_from_obs(b_actions)

        return {
            'observations': pad_observations,     
            'mask_observations': mask_observations,
            'j_observations': pad_j_observations,
            'mask_j_observations': mask_j_observations,
            'actions': pad_actions,
            'mask_actions': mask_actions,
            'logprobs': b_logprobs, 
            'returns': returns,
            'advantages': advantages
        } 

    def clear(self):
        self.idx = 0
        self.ep_lens = []

       
