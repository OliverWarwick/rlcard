''' A toy example of playing nano OFCP with random agents
'''

import rlcard
from rlcard.agents import RandomAgent
from rlcard.utils import set_global_seed

# Make environment
env = rlcard.make('nano_ofcp', config={'seed': 0})
episode_num = 1

# Set a global seed
set_global_seed(0)

# Set up agents
agent_0 = RandomAgent(action_num=env.action_num)
agent_1 = RandomAgent(action_num=env.action_num)
env.set_agents([agent_0, agent_1])

for episode in range(episode_num):

    # Generate data from the environment
    trajectories, _ = env.run(is_training=False)

    # Print out the trajectories
    print('\nEpisode {}'.format(episode))
    for ts in trajectories[0]:
        print('State: {} \nAction: {} \nReward: {} \nNext State: {} \nDone: {} \n\n'.format(ts[0], ts[1], ts[2], ts[3], ts[4]))
