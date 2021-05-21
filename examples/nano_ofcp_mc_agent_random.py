''' A toy example of playing nano OFCP with random agents
'''

import rlcard
from rlcard.agents import RandomAgent, MonteCarloAgent
from rlcard.utils import set_global_seed

# Make environment
env = rlcard.make('nano_ofcp', config={'seed': 0, 'allow_step_back': True})
episode_num = 1

# Set a global seed
set_global_seed(0)

# Set up agents
mc_agent = MonteCarloAgent(action_num=env.action_num, use_raw=False)
random_agent = RandomAgent(action_num=env.action_num)
env.set_agents([mc_agent, random_agent])

avg_reward = 0.0
for episode in range(episode_num):

    # Generate data from the environment
    trajectories, payoffs = env.mc_agent_run(is_training=False)
    avg_reward += (payoffs[0] - avg_reward) / (episode + 1)

    # # Print out the trajectories
    # print('\nEpisode {}'.format(episode))
    # for ts in trajectories[0]:
    #     print('State: {} \nAction: {} \nReward: {} \nNext State: {} \nDone: {} \n\n'.format(ts[0], ts[1], ts[2], ts[3], ts[4]))

print("Avg Reward: {}".format(avg_reward))