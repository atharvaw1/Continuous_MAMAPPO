import sys;

import torch
from torch import Type

sys.path.append('../..')
import warnings;

warnings.filterwarnings("ignore", category=Warning)
import time
from collections import deque
from copy import deepcopy

import torch.nn as nn
import torch.optim as optim

import config
from agent import GaussianActor, Critic
from utils.misc import *
from utils.memory import Buffer


def _to_dict_clip_array(dict: Dict, min: List[float], max: List[float]) -> Dict:
    return {k: np.array(v).clip(min[k], max[k]) for k, v in dict.items()}


def _array_to_dict_tensor(agents: List[str], data: Array, device: th.device, prev_ma: Dict = None,astype: Type = th.float32) -> Dict:
    if prev_ma:
        return {k: th.as_tensor([np.concatenate([d,prev_ma[k][0]])], dtype=astype).to(device) for k, d in zip(agents, data)}
    return {k: th.as_tensor([d], dtype=astype).to(device) for k, d in zip(agents, data)}

def _get_joint_obs(observations: Dict[str, Tensor]) -> Tensor:
    return th.cat(list(observations.values())).reshape(1, -1)


if __name__ == "__main__":
    args = config.parse_args()

    # Loggers
    run_name = f"{args.env}__{args.tag}__{args.seed}__{int(time.time())}__{np.random.randint(0, 100)}"
    summary_w, wandb_path = init_loggers(run_name, args)

    # Torch init and seeding 
    set_seeds(args.seed, args.th_deterministic)
    device = set_torch(args.n_cpus, args.cuda)

    # Environment setup    
    env = env_ids[args.env](args.seed, args.max_steps)

    agents = env.agent_ids
    a_low = {k: space.low for k, space in env.ma_space.items()}
    a_high = {k: space.high for k, space in env.ma_space.items()}

    # Actor-Critics setup
    actors, critics, a_optim, c_optim, buffers = {}, {}, {}, {}, {}
    params = []
    for k in agents:
        actors[k] = GaussianActor(env, agents, args.h_size, args.n_hidden).to(device)
        a_optim[k] = optim.Adam(list(actors[k].parameters()), lr=args.actor_lr, eps=1e-5)
        critics[k] = Critic(env, agents, args.h_size, args.n_hidden).to(device)
        c_optim[k] = optim.Adam(list(critics[k].parameters()), lr=args.critic_lr, eps=1e-5)
        buffers[k] = Buffer(env, agents, args.n_steps, args.max_steps, args.gamma, args.gae, args.gae_lambda, device)

    # Training metrics
    global_step, ep_count = 0, 0
    start_time = time.time()
    reward_q, cost_q = deque(maxlen=args.last_n), deque(maxlen=args.last_n)

    n_updates = args.tot_steps // args.batch_size
    for update in range(1, n_updates + 1):
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / n_updates
            for a_opt, c_opt in zip(a_optim.values(), c_optim.values()):
                a_opt.param_groups[0]["lr"] = frac * args.actor_lr
                c_opt.param_groups[0]["lr"] = frac * args.critic_lr

        # Environment reset
        prev_ma_action = {k: th.empty((args.n_envs, env.ma_space[k].shape[0])) for k in agents}
        observation = _array_to_dict_tensor(agents, env.reset(), device, prev_ma_action)

        ma_observation = deepcopy(observation)
        j_observation = {}
        ma_action = {k: th.empty((args.n_envs, env.ma_space[k].shape[0])) for k in agents}
        ma_mean = deepcopy(ma_action)
        ma_std = deepcopy(ma_action)
        ma_logprob = {k: th.zeros(args.n_envs) for k in agents}
        ma_step, ma_gamma, ma_reward = {}, {}, {}

        ma_value = deepcopy(ma_logprob)
        ma_done = {k: th.ones(args.n_envs) for k in agents}
        a_h, a_h_ = [{k: th.zeros((1, args.h_size)) for k in agents} for _ in range(2)]
        c_h, c_h_ = [{k: th.zeros((1, args.h_size)) for k in agents} for _ in range(2)]

        ma_count = {k: th.zeros(args.n_envs) for k in agents}

        ep_reward, ep_cost, ep_step, ep_macro = 0, 0, 0, 0
        while any(np.sum(list(ma_count.values())) < args.n_steps):
            global_step += 1 * args.n_envs
            ep_step += 1
            with th.no_grad():
                for k in agents:
                    if ma_done[k]:
                        if k == agents[0]: ep_macro += 1

                        ma_step[k], ma_reward[k] = [th.zeros(1) for _ in range(2)]
                        ma_gamma[k] = th.ones(1)

                        ma_count[k] += 1
                        ma_observation[k] = observation[k]
                        (
                            ma_action[k],
                            ma_logprob[k],
                            _,
                            ma_mean[k],
                            ma_std[k],
                            a_h_[k]
                        ) = actors[k].get_action(observation[k], h=a_h[k])

                for k in agents:
                    if ma_done[k]:
                        j_observation[k] = _get_joint_obs(ma_observation)
                        ma_value[k], c_h_[k] = critics[k](j_observation[k], h=c_h[k])


            observation_, reward, done, info = env.step(_to_dict_clip_array(ma_action, a_low, a_high))

            ma_done = _array_to_dict_tensor(agents, info['ma_done'], device)
            reward = _array_to_dict_tensor(agents, reward, device)
            cost = _array_to_dict_tensor(agents, info['cost'], device)
            done = _array_to_dict_tensor(agents, done, device)

            for k in agents:
                ma_step[k] += 1
                ma_reward[k] += args.gamma ** (ma_step[k] - 1) * reward[k]
                ma_gamma[k] = args.gamma ** (ma_step[k] - 1)

                if ma_done[k]:
                    buffers[k].store(
                        ma_observation[k],
                        j_observation[k],
                        ma_action[k],
                        ma_mean[k],
                        ma_std[k],
                        ma_logprob[k],
                        ma_reward[k],
                        ma_value[k],
                        done[k]
                    )
                    a_h[k] = a_h_[k]
                    c_h[k] = c_h_[k]

            prev_ma_action = ma_action
            observation = _array_to_dict_tensor(agents, observation_,device,prev_ma_action)

            # Consider the metrics in the first agent as its fully cooperative
            ep_reward += reward[agents[0]].numpy()
            ep_cost += np.sum(info['cost'])

            if all(list(done.values())):
                ep_count += 1

                reward_q.append(ep_reward)
                cost_q.append(ep_cost)

                # Should record one for each environment? Don't know yet
                record = {
                    'Global_Step': global_step,
                    'Reward': ep_reward,
                    'Avg_Reward': np.mean(reward_q),
                    'Cost': ep_cost,
                    'Avg_Cost': np.mean(cost_q),
                    'Macro': ep_macro
                }

                if args.tb_log: summary_w.add_scalars('Training', record, global_step)
                if args.wandb_log: wandb.log(record)

                # Save model weights at training intervals
                if args.model_checkpoint and global_step % args.model_checkpoint_interval == 0:
                    for k in agents:
                        torch.save(actors[k].state_dict(), f"outputs/actor_{k}_model_checkpoint_step_{global_step}.pt")
                        torch.save(critics[k].state_dict(),
                                   f"outputs/critic_{k}_model_checkpoint_step_{global_step}.pt")

                if args.verbose:
                    print(f"E: {ep_count},\n\t "
                          f"Reward: {record['Reward']},\n\t "
                          f"Avg_Reward: {record['Avg_Reward']},\n\t "
                          f"Avg_Cost: {record['Avg_Cost']},\n\t "
                          f"Macro: {ep_macro},\n\t "
                          f"Global_Step: {global_step},\n\t "
                          )

                ep_step, ep_reward, ep_cost, ep_macro = 0, 0, 0, 0
                prev_ma_action = {k: th.empty((args.n_envs, env.ma_space[k].shape[0])) for k in agents}
                observation = _array_to_dict_tensor(agents, env.reset(), device, prev_ma_action)
                a_h = {k: th.zeros((1, args.h_size)) for k in agents}
                c_h = {k: th.zeros((1, args.h_size)) for k in agents}
                ma_done = {k: th.ones(args.n_envs) for k in agents}

        with th.no_grad():
            for k in agents:
                if ma_done[k]:
                    ma_observation[k] = observation[k]
                j_observation = _get_joint_obs(ma_observation)

                for k in agents:
                    value_, _ = critics[k](j_observation, c_h[k])
                    buffers[k].compute_mc(value_.reshape(-1))

        # Optimize the policy and value networks
        for k in agents:
            b = buffers[k].sample()
            buffers[k].clear()

            for epoch in range(args.n_epochs):
                _, logprob, entropy, _, _, _ = actors[k].get_action(b['observations'], b['actions'])
                entropy_loss = entropy.mean()

                logratio = logprob - b['logprobs']
                ratio = logratio.exp()

                mb_advantages = b['advantages']
                if args.norm_adv:
                    mb_advantages = normalize(mb_advantages)

                actor_loss = mb_advantages * ratio
                actor_clip_loss = mb_advantages * th.clamp(ratio, 1 - args.clip, 1 + args.clip)
                actor_loss = th.min(actor_loss, actor_clip_loss).mean()

                actor_loss = -actor_loss - args.ent_coef * entropy_loss

                a_optim[k].zero_grad(True)
                actor_loss.backward()
                nn.utils.clip_grad_norm_(actors[k].parameters(), args.max_grad_norm)
                a_optim[k].step()

                values, _ = critics[k](b['j_observations'])
                values = values.squeeze()

                critic_loss = 0.5 * ((values - b['returns']) ** 2).mean()
                critic_loss = critic_loss * args.v_coef

                c_optim[k].zero_grad(True)
                critic_loss.backward()
                nn.utils.clip_grad_norm_(critics[k].parameters(), args.max_grad_norm)
                c_optim[k].step()

                with th.no_grad():  # To early break from updates -> not used here
                    if args.target_kl is not None:
                        approx_kl = ((ratio - 1) - logratio).mean()
                        if approx_kl > args.target_kl:
                            break

        sps = int(global_step / (time.time() - start_time))
        if args.tb_log: summary_w.add_scalar('Training/SPS', sps, global_step)
        if args.wandb_log: wandb.log({'SPS': sps})

        if global_step > args.tot_steps: break

    try:
        if args.tb_log: summary_w.close()
        if args.wandb_log:
            wandb.finish()
            if args.wandb_mode == 'offline':
                import subprocess

                subprocess.run(['wandb', 'sync', wandb_path])
        env.close()
    except:
        pass
