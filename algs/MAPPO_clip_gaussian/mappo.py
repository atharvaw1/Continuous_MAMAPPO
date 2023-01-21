import sys; sys.path.append('../..')
import warnings; warnings.filterwarnings("ignore", category=Warning) 
import time
from collections import deque
import torch.nn as nn
import torch.optim as optim

import config
from agent import GaussianActor, Critic
from utils.misc import *
from utils.memory import Buffer

def _to_dict_clip_array(dict: Dict, min: List[float], max: List[float]) -> Dict:
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
    actors, a_optim, buffers = {}, {}, {}
    params = []
    for k in agents:
        actors[k] = GaussianActor(env, agents, args.h_size, args.n_hidden).to(device)
        a_optim[k] = optim.Adam(list(actors[k].parameters()), lr=args.actor_lr, eps=1e-5)
        buffers[k] = Buffer(env, agents, args.n_steps, args.max_steps, args.gamma, args.gae, args.gae_lambda, device)

    critic = Critic(env, agents, args.h_size, args.n_hidden).to(device)
    c_optim = optim.Adam(list(critic.parameters()), lr=args.critic_lr, eps=1e-5)

    # Training metrics
    global_step, ep_count = 0, 0
    start_time = time.time()
    reward_q, cost_q = deque(maxlen=args.last_n), deque(maxlen=args.last_n)

    n_updates = args.tot_steps // args.batch_size
    for update in range(1, n_updates + 1):
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / n_updates
            for opt in a_optim.values(): 
                opt.param_groups[0]["lr"] = frac * args.actor_lr
            c_optim.param_groups[0]["lr"] = frac * args.critic_lr

        # Environment reset
        observation = _array_to_dict_tensor(agents, env.reset(), device)
        action, logprob, mean, std = [{k: 0 for k in agents} for _ in range(4)]
        a_h = {k: th.zeros((1, args.h_size)) for k in agents} 
        c_h =  th.zeros((1, args.h_size))
        ep_reward, ep_cost = 0, 0

        for step in range(args.n_steps):
            global_step += 1 * args.n_envs

            with th.no_grad():
                for k in agents:
                    (
                        action[k], 
                        logprob[k], 
                        _, 
                        mean[k], 
                        std[k],
                        a_h[k]
                    ) = actors[k].get_action(observation[k], h=a_h[k])
                    
                j_observation = _get_joint_obs(observation)
                s_value, c_h = critic(j_observation, h=c_h)  

            # env.render()

            observation_, reward, done, info = env.step(_to_dict_clip_array(action, a_low, a_high))    
            
            reward = _array_to_dict_tensor(agents, reward, device)
            cost = _array_to_dict_tensor(agents, info['cost'], device)
            done = _array_to_dict_tensor(agents, done, device)
    
            for k in agents:               
                buffers[k].store(
                    observation[k], 
                    j_observation,
                    action[k], 
                    mean[k], 
                    std[k], 
                    logprob[k], 
                    reward[k], 
                    s_value, 
                    done[k]   
                )
                
            observation = _array_to_dict_tensor(agents, observation_, device)

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
                    'Avg_Cost': np.mean(cost_q)
                }

                if args.tb_log: summary_w.add_scalars('Training', record, global_step) 
                if args.wandb_log: wandb.log(record)

                if args.verbose:
                    print(f"E: {ep_count},\n\t "
                        f"Reward: {record['Reward']},\n\t "
                        f"Avg_Reward: {record['Avg_Reward']},\n\t "
                        f"Avg_Cost: {record['Avg_Cost']},\n\t "
                        f"Global_Step: {global_step},\n\t "
                    )

                ep_step, ep_reward, ep_cost = 0, 0, 0
                observation = _array_to_dict_tensor(agents, env.reset(), device)
                a_h = {k: th.zeros((1, args.h_size)) for k in agents} 
                c_h = th.zeros((1, args.h_size))

        with th.no_grad():
            j_observation = _get_joint_obs(observation)
            value_, _ = critic(j_observation, c_h)  
            for k in agents:
                buffers[k].compute_mc(value_.reshape(-1))
                
        # Optimize the policy and value networks
        for k in agents:
            b = buffers[k].sample() 
            buffers[k].clear()

            for epoch in range(args.n_epochs):

                a_h = {k: th.zeros((1, args.h_size)) for k in agents} 
                c_h = th.zeros((1, args.h_size))

                _, logprob, entropy, _, _, _ = actors[k].get_action(b['observations'], b['actions'])
                entropy_loss = entropy.mean()     # ([10])
            
               
                logratio = logprob - b['logprobs']  # (10, 50)
                ratio = logratio.exp()  # (10, 50)
              
                mb_advantages = b['advantages'] # (10, 50) -> it's meh to norm over all the episodes
                if args.norm_adv:
                    mb_advantages = normalize(mb_advantages)
        
                actor_loss = mb_advantages * ratio
                actor_clip_loss = mb_advantages * th.clamp(ratio, 1 - args.clip, 1 + args.clip)
                actor_loss = th.min(actor_loss, actor_clip_loss).mean()   # ([10])            
                    
                actor_loss = -actor_loss - args.ent_coef * entropy_loss

                a_optim[k].zero_grad(True)
                actor_loss.backward()
                nn.utils.clip_grad_norm_(actors[k].parameters(), args.max_grad_norm)
                a_optim[k].step()

                values, _ = critic(b['j_observations'])
                values = values.squeeze()
                
                critic_loss = 0.5 * ((values - b['returns']) ** 2).mean()
                critic_loss = critic_loss * args.v_coef
               
                c_optim.zero_grad(True)
                critic_loss.backward()
                nn.utils.clip_grad_norm_(critic.parameters(), args.max_grad_norm)
                c_optim.step()

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

