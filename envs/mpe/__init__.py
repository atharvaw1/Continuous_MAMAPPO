import os, sys
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)   # .../mpe
sys.path.append(parent)

from gym.envs.registration import register
import envs.mpe.scenarios as scenarios

# Multiagent envs
# ----------------------------------------

_particles = {
    "simple_reference": "SimpleReference-v0",
    "simple_speaker_listener": "SimpleSpeakerListener-v0",
    "simple_spread": "SimpleSpread-v0",
    "simple_tag": "SimpleTag-v0",
}

for scenario_name, gymkey in _particles.items():
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    #world = scenario.make_world()

    # Registers multi-agent particle environments:
    register(
        gymkey,
        entry_point="mpe.environment:MultiAgentEnv",
        kwargs={
            #"world": world,
            "reset_callback": scenario.reset_world,
            "reward_callback": scenario.reward,
            "observation_callback": scenario.observation,
        },
    )