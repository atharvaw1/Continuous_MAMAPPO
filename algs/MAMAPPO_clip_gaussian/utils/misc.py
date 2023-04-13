import argparse
import os
from typing import Tuple, Optional, Dict, Union, List, Type

import gym
import numpy as np
import torch as th
import wandb
from torch.utils.tensorboard import SummaryWriter

from envs.ma_mpe_wrappers import MaSpreadWrapper, MaReferenceWrapper#, MaSpeakerWrapper, MaTagWrapper

Tensor = th.Tensor
Array = np.array

env_ids = {
    'MaSpreadWrapper': MaSpreadWrapper,
    'MaReferenceWrapper': MaReferenceWrapper,
    #'MaSpeakerWrapper': MaSpeakerWrapper,
    #'MaTagWrapper': MaTagWrapper,
}

@th.jit.script
def normalize(x: Tensor) -> Tensor:
    return (x - x.mean()) / (x.std() + 1e-8)
    
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def set_seeds(seed: int, deterministic: Optional[bool]):
    np.random.seed(seed)
    th.manual_seed(seed)
    th.backends.cudnn.deterministic = deterministic

def set_torch(n_cpus: int, cuda: bool) -> th.device:
    th.set_num_threads(n_cpus)
    return th.device("cuda" if th.cuda.is_available() and cuda else "cpu")

def init_loggers(run_name: str, args: Dict[str, Union[int, float, List]]) -> Tuple[SummaryWriter, str]:
    summary_w, wandb_path = None, None
    if args.wandb_log:
        wandb_path = wandb.init(
            name=run_name,
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            mode=args.wandb_mode,
            save_code=args.wandb_code,
            config=vars(args)
            )
        wandb_path =  os.path.split(wandb_path.dir)[0]
    if args.tb_log:
        summary_w = SummaryWriter(f"log/{run_name}")
        summary_w.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
        )
    return summary_w, wandb_path

