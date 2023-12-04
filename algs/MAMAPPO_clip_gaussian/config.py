"""Parser

"""
import argparse

from utils.misc import str2bool

def parse_args():
    parser = argparse.ArgumentParser()

    # Logging    
    parser.add_argument("--verbose", type=str2bool, default=True, help="Log/print output")   
    parser.add_argument("--tb-log", type=str2bool, default=False, help="Tensorboard log")    
    parser.add_argument("--wandb-log", type=str2bool, default=True, help="Wandb log")
    parser.add_argument("--tag", type=str, default='MAMAPPO_clip_gaussian', help="Training tag")

    # Environment
    # Cooperative: ['MaReferenceWrapper', 'MaSpeakerWrapper', 'MaSpreadWrapper'], 
    # Competitive: ['MaTagWrapper']
    parser.add_argument("--env", type=str, default="MaSpreadWrapper", help="the name of the gym environment") 
    parser.add_argument("--n-envs", type=int, default=1, help="n° of envs") 
    parser.add_argument("--seed", type=int, default=1, help="seed of the experiment")

    #parser.add_argument("--norm-obs", type=str2bool, default=True, help="Normalize observations")
    #parser.add_argument("--norm-rew", type=str2bool, default=False, help="Normalize rewards")

    # Environment setup
    parser.add_argument("--max-steps", type=int, default=50, help="Max n° of steps per episode")

    # Experiment
    parser.add_argument("--n-steps", type=int, default=500, help="the number of steps between policy updates")
    parser.add_argument("--tot-steps", type=int, default=4000000, help="total timesteps of the experiments")

    # Algorithm 
    parser.add_argument("--clip", type=float, default=0.2, help="the surrogate clipping coefficient")
    parser.add_argument("--clip-v", type=str2bool, default=False,
        help="Toggles wheter or not to use a clipped loss for the value function.")
    parser.add_argument("--target-kl", type=float, default=None, help="the target KL divergence threshold")
    parser.add_argument("--gamma", type=float, default=0.9, help="the discount factor gamma")
    parser.add_argument("--gae", type=str2bool, default=True, help="Use gae")
    parser.add_argument("--gae-lambda", type=float, default=0.95, help="the lambda for the gae")
    parser.add_argument("--ent-coef", type=float, default=1e-3, help="coefficient of the entropy")
    parser.add_argument("--v-coef", type=float, default=0.5, help="coefficient of the value function")

    # Update
    parser.add_argument("--actor-lr", type=float, default=3e-4, help="the learning rate of the actors' optimizers")
    parser.add_argument("--critic-lr", type=float, default=1e-3, help="the learning rate of the critic's optimizer")
    parser.add_argument("--anneal-lr", type=str2bool, default=True,
        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--pi-epochs", type=int, default=10, help="the epochs to update the policy")
    parser.add_argument("--vf-epochs", type=int, default=30, help="the epochs to update the policy")

    parser.add_argument("--norm-adv", type=str2bool, default=False,
        help="Toggles advantages normalization")
    parser.add_argument("--max-grad-norm", type=float, default=1.,
        help="the maximum norm for the gradient clipping")
    parser.add_argument("--h-size", type=int, default=64, help="the size of the dnn")
    parser.add_argument("--n_hidden", type=int, default=2, help="n° of hidden layers")
    
    # Metrics
    parser.add_argument("--last-n", type=int, default=100, help="Average metrics over this time horizon")

    # wandb
    parser.add_argument("--wandb-project-name", type=str, default="mamappo_fixed", help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default="wandile-a", help="the entity (team) of wandb's project")
    parser.add_argument("--wandb-mode", type=str, default="online", 
        help="online or offline wadb mode. if offline,, we'll try to sync immediately after the run")
    parser.add_argument("--wandb-code", type=str2bool, default=False, 
        help="Save code in wandb")

    # Torch
    parser.add_argument("--n-cpus", type=int, default=4, help="N° of cpus/max threads for process")
    parser.add_argument("--th-deterministic", type=str2bool, default=True, 
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=str2bool, default=True, 
        help="if toggled, cuda will be enabled by default")

    args = parser.parse_args()
    args.batch_size = int(args.n_envs * args.n_steps)

    return args