import os, sys, inspect
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)   # .../mpe
sys.path.append(parent)

from gym.envs.registration import register
import envs.mpe.scenarios as scenarios

# Multiagent envs
# ----------------------------------------

_particles = {
    #"multi_speaker_listener": "MultiSpeakerListener-v0",
    #"simple_adversary": "SimpleAdversary-v0",
    #"simple_crypto": "SimpleCrypto-v0",
    #"simple_push": "SimplePush-v0",
    #"simple_reference": "SimpleReference-v0",
    #"simple_speaker_listener": "SimpleSpeakerListener-v0",
    "simple_spread": "SimpleSpread-v0",
    #"simple": "Simple-v0",
    #"simple_tag": "SimpleTag-v0",
    #"simple_world_comm": "SimpleWorldComm-v0",
    #"climbing_spread": "ClimbingSpread-v0",
}

for scenario_name, gymkey in _particles.items():
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    # We do the following in the wrappers script to pass additional parameters
    # world = scenario.make_world() 

    # Registers multi-agent particle environments:
    register(
        gymkey,
        entry_point="mpe.environment:MultiAgentEnv",
        kwargs={
            #"world": world,    # So we also have to remove it from here
            "reset_callback": scenario.reset_world,
            "reward_callback": scenario.reward,
            "observation_callback": scenario.observation,
            "info_callback": scenario.collision
        },
    )
'''
# Registers the custom double spread environment:
for N in range(2, 11, 2):
    scenario_name = "simple_doublespread"
    gymkey = f"DoubleSpread-{N}ag-v0"
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    world = scenario.make_world(N)

    register(
        gymkey,
        entry_point="mpe.environment:MultiAgentEnv",
        kwargs={
            "world": world,
            "reset_callback": scenario.reset_world,
            "reward_callback": scenario.reward,
            "observation_callback": scenario.observation,
        },
    )
'''