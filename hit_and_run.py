# A bot trained using reinforcement learning
# https://github.com/SoyGema/Startcraft_pysc2_minigames
# https://gamescapad.es/building-bots-in-starcraft-2-for-psychologists/#installation
# had to add minimap to pysc2\maps\mini_games.py
# /home/jeff/anaconda3/envs/starcraft_test/lib/python3.6/site-packages/pysc2/
from pysc2.agents import base_agent
from pysc2.env import sc2_env
from pysc2.lib import actions, features, units
from absl import app

class HitAndRunAgent(base_agent.BaseAgent):
    def step(self, obs):
        super(HitAndRunAgent, self).step(obs)
        
        return actions.FunctionCall(actions.FUNCTIONS.no_op.id, [])

def main(unused_argv):
    del unused_argv
    agent = HitAndRunAgent()
    try:
        while True:
            with sc2_env.SC2Env(
                map_name="HitAndRun",
                players=[sc2_env.Agent(sc2_env.Race.zerg),
                        sc2_env.Bot(sc2_env.Race.random,
                                    sc2_env.Difficulty.very_easy)],
                agent_interface_format=features.AgentInterfaceFormat(
                    feature_dimensions=features.Dimensions(screen=64, minimap=32),
                    use_feature_units=True),
                #   step_mul=16,
                game_steps_per_episode=0,
                visualize=True) as env:
                
                agent.setup(env.observation_spec(), env.action_spec())
                
                timesteps = env.reset()
                agent.reset()
                
                while True:
                    step_actions = [agent.step(timesteps[0])]
                    if timesteps[0].last():
                        break
                    timesteps = env.step(step_actions)
    except KeyboardInterrupt:
        pass

if __name__ == '__main__':
    app.run(main)