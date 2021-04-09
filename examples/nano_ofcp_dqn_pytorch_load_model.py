''' An example of loading pre-trained NFSP model on Leduc Holdem
'''
import os
import torch

import rlcard
from rlcard.agents.dqn_agent_pytorch import DQNAgent
from rlcard.agents.random_agent import RandomAgent
from rlcard.utils.utils import set_global_seed, tournament

# Set a global seed
set_global_seed(0)

# Make environment
env = rlcard.make('nano_ofcp', config={'seed': 0})

# Load up the class of the DQN agent, and then we can populate with the weights from the checkpoint.
agent = DQNAgent(scope='dqn',
                 action_num=env.action_num,
                 state_shape=env.state_shape,
                 mlp_layers=[64, 64],
                 device=torch.device('cpu')
                 )

random_agent = RandomAgent(action_num=env.action_num)

# We have a pretrained model here. Change the path for your model.
# check_point_path = os.path.join('rlcard/examples/models/nano_dqn_pytorch/model.pth')

# TODO: Fix the hard coding going on here.
checkpoint = torch.load('/Users/student/rlcard/models/nano_dqn_pytorch/model.pth')
agent.load(checkpoint)

# Evaluate the performance. Play with random agents.
evaluate_num = 100
env.set_agents([agent, random_agent])
reward = tournament(env, evaluate_num)[0]
print('Average reward against random agent: ', reward)