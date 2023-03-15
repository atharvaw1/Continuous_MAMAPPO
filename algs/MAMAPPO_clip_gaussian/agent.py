import gym
from torch.distributions import Normal
import torch.nn as nn
import torch.nn.functional as F

from utils.misc import *


def Linear(input_dim, output_dim, act_fn='tanh', init_weight_uniform=True):
    """
    Creat a linear layer.

    Parameters
    ----------
    input_dim : int
        The input dimension.
    output_dim : int
        The output dimension.
    act_fn : str
        The activation function.
    init_weight_uniform : bool
        Whether uniformly sample initial weights.
    """
    gain = th.nn.init.calculate_gain(act_fn)
    fc = th.nn.Linear(input_dim, output_dim)
    if init_weight_uniform:
        nn.init.xavier_uniform_(fc.weight, gain=gain)
    else:
        nn.init.xavier_normal_(fc.weight, gain=gain)
    nn.init.constant_(fc.bias, 0.00)
    return fc


class GaussianActor(nn.Module):
    def __init__(self, env: gym.Env, agents: List[str], h_size: int, n_hidden: int):
        super().__init__()

        o_space, a_space = env.observation_space[agents[0]], env.ma_space[agents[0]]

        self.hidden_1 = Linear(int(np.prod(o_space.shape)+ np.prod(a_space.shape)), h_size)
        # self.hidden_1 = Linear(int(np.prod(o_space.shape)), h_size)
        self.gru_1 = nn.GRU(h_size, h_size, batch_first=True)
        self.hidden_2 = Linear(h_size, h_size)
        self.output = Linear(h_size, int(np.prod(a_space.shape)), act_fn='tanh')

        # self.logstd = nn.Parameter(-th.ones(int(np.prod(a_space.shape))))
        self.logstd = nn.Parameter(-th.zeros(int(np.prod(a_space.shape))))

    def forward(self, x, h=None):
        x = F.tanh(self.hidden_1(x))
        x, h_ = self.gru_1(x, h)
        x = F.tanh(self.hidden_2(x))
        x = F.tanh(self.output(x))
        return x, h_

    def get_action(self, x, y=None, h=None):
        mean, h_ = self.forward(x, h)
        logstd = self.logstd.expand_as(mean)
        std = th.exp(logstd)
        prob = Normal(mean, std)
        if y is None:
            y = prob.sample()
        return y, prob.log_prob(y).sum(-1), prob.entropy().sum(-1), mean, std, h_


class Critic(nn.Module):
    def __init__(self, env, agents: List[str], h_size: int, n_hidden: int):
        super().__init__()

        o_size = (env.observation_space[agents[0]].shape[0] + env.ma_space[agents[0]].shape[0]) * len(agents)
        # o_size = env.observation_space[agents[0]].shape[0] * len(agents)
        self.hidden_1 = Linear(o_size, h_size)
        self.gru_1 = nn.GRU(h_size, h_size, batch_first=True)
        self.hidden_2 = Linear(h_size, h_size)
        self.output = Linear(h_size, 1, act_fn='linear')

    def forward(self, x, h=None):
        x = F.tanh(self.hidden_1(x))
        x, h_ = self.gru_1(x, h)
        x = F.tanh(self.hidden_2(x))
        x = self.output(x)
        return x, h_
