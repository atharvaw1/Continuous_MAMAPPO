import sys; sys.path.append('../..')
import warnings; warnings.filterwarnings("ignore", category=Warning) 
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
    return {k: np.array(v) for k, v in dict.items()}
    return {k: np.array(v).clip(min[k], max[k]) for k, v in dict.items()}

def _array_to_dict_tensor(agents: List[str], data: Array, device: th.device, astype: Type = th.float32) -> Dict:
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
        observation = _array_to_dict_tensor(agents, env.reset(), device)

        ma_observation = deepcopy(observation)
        j_observation = {}
        ma_action = {k: th.zeros((args.n_envs, env.ma_space[k].shape[0])) for k in agents}
        ma_mean = deepcopy(ma_action)
        ma_std = deepcopy(ma_action)
        ma_logprob = {k: th.zeros(args.n_envs) for k in agents}
        ma_step, ma_gamma, ma_reward = {}, {}, {}

        ma_value = deepcopy(ma_logprob)
        ma_done = {k: th.ones(args.n_envs) for k in agents}
        a_h, a_h_ = [{k: th.zeros((1, args.h_size)) for k in agents} for _ in range(2)]
        c_h, c_h_ =  [{k: th.zeros((1, args.h_size)) for k in agents} for _ in range(2)]

        ma_count = {k: th.zeros(args.n_envs) for k in agents}

        ep_reward, ep_cost, ep_step, ep_macro = 0, 0, 0, 0
        #while any(np.sum(list(ma_count.values())) < args.n_steps):
        while all(np.array(list(ma_count.values())) < args.n_steps):
            #print(np.array(list(ma_count.values())))
            global_step += 1 * args.n_envs
            ep_step += 1

            with th.no_grad():
                for k in agents:
                    if ma_done[k]:
                        if k == agents[0]: ep_macro += 1
                        
                        # reset ma reward / discount
                        ma_reward[k] = th.zeros(1)
                        ma_step[k] = th.ones(1) # Should be 1 in order to have the first action to be discounted by gamma
                        ma_gamma[k] = th.ones(1)

                        ma_observation[k] = deepcopy(observation[k])
                        ma_observation[k] = th.cat([ma_observation[k], ma_action[k]], dim=-1)
                        (
                            ma_action[k], 
                            ma_logprob[k], 
                            _, 
                            ma_mean[k], 
                            ma_std[k],
                            a_h_[k]
                        ) = actors[k].get_action(ma_observation[k], h=a_h[k])
                        
                # We need all the updated ma_observations for the joint one
                for k in agents:
                    if ma_done[k]:     
                        j_observation[k] = _get_joint_obs(ma_observation)
                        ma_value[k], c_h_[k] = critics[k](j_observation[k], h=c_h[k])  

            #print(f"Agent 0 pos: {np.array(list(ma_observation.values()))[0][0][2:4]}")
            #print(list(ma_action.values())[0])

            #print(f"Agent 1 pos: {np.array(list(ma_observation.values()))[1][0][2:4]}")
            #print(list(ma_action.values())[1])

            #print(f"Agent 2 pos: {np.array(list(ma_observation.values()))[2][0][2:4]}")
            #print(list(ma_action.values())[2])
            #print(f"Position agent 0: {np.array(list(observation.values()))[0][0][2:4]}")
            #print(f"Position agent 1: {np.array(list(observation.values()))[1][0][2:4]}")
            #print(f"Position agent 2: {np.array(list(observation.values()))[2][0][2:4]}")

            observation_, reward, done, info = env.step(_to_dict_clip_array(ma_action, a_low, a_high))    
           
            ma_done = _array_to_dict_tensor(agents, info['ma_done'], device)
            reward = _array_to_dict_tensor(agents, reward, device)
            cost = _array_to_dict_tensor(agents, info['cost'], device)
            done = _array_to_dict_tensor(agents, done, device)

            observation = _array_to_dict_tensor(agents, deepcopy(observation_), device)
            #env.render()
           
            for k in agents:      
                ma_step[k] += 1
                ma_reward[k] += args.gamma ** (ma_step[k] - 1) * reward[k]  
                ma_gamma[k] = args.gamma ** (ma_step[k] - 1)

                if ma_done[k] or all(list(done.values())):
                    ma_count[k] += 1

                    buffers[k].store(
                        ma_observation[k], 
                        j_observation[k],
                        ma_action[k], 
                        ma_mean[k], 
                        ma_std[k], 
                        ma_logprob[k], 
                        ma_reward[k], 
                        ma_gamma[k],
                        ma_value[k], 
                        done[k]   
                    )

                    a_h[k] = deepcopy(a_h_[k])
                    c_h[k] = deepcopy(c_h_[k])
           

            # Consider the metrics in the first agent as its fully cooperative
            ep_reward += reward[agents[0]].numpy()
            ep_cost += np.sum(info['cost'])

            if all(list(done.values())):
                for b in buffers.values():
                    b._store_ep_len()
                    
                ep_count += 1
               
                reward_q.append(ep_reward)
                cost_q.append(ep_cost)

                # Should record one for each environment? Don't know yet
                record = {
                    'Global_Step': global_step,
                    'Reward': ep_reward,
                    'Avg_Reward': np.mean(reward_q),
                    'Cost': ep_cost,
                    'Avg_Cost': np.mean(cost_q)
                }

                if args.tb_log: summary_w.add_scalars('Training', record, global_step) 
                if args.wandb_log: wandb.log(record)

                if args.verbose:
                    print(f"E: {ep_count},\n\t "
                        f"Reward: {record['Reward']},\n\t "
                        f"Avg_Reward: {record['Avg_Reward']},\n\t "
                        f"Avg_Cost: {record['Avg_Cost']},\n\t "
                        f"Macro: {ep_macro},\n\t "
                        f"Global_Step: {global_step},\n\t "
                    )
                
                ep_step, ep_reward, ep_cost, ep_macro = 0, 0, 0, 0
                observation = _array_to_dict_tensor(agents, deepcopy(env.reset()), device)
                a_h = {k: th.zeros((1, args.h_size)) for k in agents} 
                c_h = {k: th.zeros((1,  args.h_size)) for k in agents} 
                ma_done = {k: th.ones(args.n_envs) for k in agents}

        if not all(list(done.values())):
            for b in buffers.values():
                b._store_ep_len()

        with th.no_grad():
            for k in agents:
                if ma_done[k]:
                    ma_observation[k] = th.cat([observation[k], ma_action[k]], dim=-1)

            j_observation = _get_joint_obs(ma_observation)

            for k in agents:
                value_, _ = critics[k](j_observation, c_h[k])  
                buffers[k].compute_mc(value_.reshape(-1))

        print("Updating")     
        # Optimize the policy and value networks
        for k in agents:
            b = buffers[k].sample() 
            buffers[k].clear()
            
            #if not b['logprobs'].shape[-1] <= 1:    # We can't train with just 1 sample, it gives NaN errors
           
            #quit()
            '''
            'observations': pad_observations,     
            'mask_observations': mask_observations,
            'j_observations': pad_j_observations,
            'mask_j_observations': mask_j_observations,
            'actions': self.b_actions,
            'mask_actions': mask_actions,
            'logprobs': self.b_logprobs, 
            'returns': self.returns,
            'advantages': self.advantages
            '''
            for epoch in range(args.pi_epochs):
                _, logprob, entropy, _, _, _ = actors[k].get_action(b['observations'], b['actions'])

                entropy_loss = entropy[b['mask_actions']].mean()     
                logratio = logprob[b['mask_actions']] - b['logprobs']  
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

                with th.no_grad():  # To early break from updates -> not used here
                    if args.target_kl is not None:
                        approx_kl = ((ratio - 1) - logratio).mean()
                        if approx_kl > args.target_kl:
                            print(f"Policy updates break at epoch: {epoch}")
                            break
           
            if args.wandb_log: 
               
                wandb.log({f'Actor loss {k}': float(actor_loss), f'Actor std {k} - Action 0': float(th.exp(actors[k].logstd[0])), f'Actor std {k} - Action 1': float(th.exp(actors[k].logstd[1])), f'Actor entropy {k}': float(entropy_loss)})


            for epoch in range(args.vf_epochs):
                values, _ = critics[k](b['j_observations'])
                values = values.squeeze()[b['mask_j_observations']]
              
                critic_loss = 0.5 * ((values - b['returns']) ** 2).mean()
                critic_loss = critic_loss * args.v_coef
               
                c_optim[k].zero_grad(True)
                critic_loss.backward()
                nn.utils.clip_grad_norm_(critics[k].parameters(), args.max_grad_norm)
                c_optim[k].step()
               
            if args.wandb_log: wandb.log({f'Critic loss {k}': float(critic_loss)})

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

