# Multi-Agent Reinforcement Learning with Continuous Macro-Actions

This repository contains:
- Macro action wrappers for petting zoo MPE environments
- Algorithms for Continuous Macro-Action Multi-Agent Proximal Policy Optimization (MAMAPPO)

## Macro Action wrappers for MPE
Using macro-actions for the agents' policy gives some abstraction to the agent and should make it easier to learn sub-goals and actions in a sample efficient way. There is a lot of flexibility in the type of macro-action to choose and also the low-level controller policy that the macro-action will follow. 

The original MPE environments use 5 primitive actions for movement. UP, DOWN, LEFT, RIGHT, NO-ACTION, and these are fairly consistent with all the environments. Some environments have  an additional communication element where agents can send a communication signal to other agents for cooperation. This communication signal can be a discrete value from 1-10.

The approach in this project was to use 2-dimensional x and y coordinates for the macro-actions and a low-level policy that will alternate between discrete x and y movements. This was chosen as it is fairly intuitive to understand and can be easily interpreted by a human. The agent then has to choose sub-goals that progressively get closer to the target goal while trying to avoid any actions that have negative rewards (like bumping into other agents or adversaries). An example action output would be (3,5), now the low-level policy will look at the agent's current position and create primitive actions that take it towards the given x,y coordinate. So if the agent is at (2,2) the primitive action output would be RIGHT->UP->RIGHT->RIGHT. For environments with communication,  the agent would output a single integer between -1 to 1 that would then be transformed into the vector to send to other agents.  The number of macro-action steps would be counted in the environment and once all agents are done then the macro-action is considered done and the agent will have to give a new macro-action. 

Another change was to remove negative rewards (collisions between agents) from the environment and to have a separate cost estimate that would be used for multi-objective optimization. This would avoid the problem of reward shaping and makes sure we can maximize rewards and minimize costs in a multi-objective  or constrained RL algorithm. 

## Continuous Macro-Action Multi-Agent Proximal Policy Optimization (MAMAPPO)

Macro-action-based MAPPO is more challenging than primitive MAPPO as it is hard to determine the agent's trajectories for updates and when to consider a macro-action terminated. This is discussed in detail in the paper [https://github.com/atharvaw1/MA_mpe/blob/master/Continuous_Macro_Action_Learning_in_Multi_Agent_RL_Environments.pdf](https://github.com/atharvaw1/Continuous_MAMAPPO/blob/master/Multi_Agent_Reinforcement_Learning_with_Continuous_Macro_Actions.pdf)

