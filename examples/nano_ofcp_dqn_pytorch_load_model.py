''' An example of loading pre-trained NFSP model on Leduc Holdem
'''
import os
import torch

import rlcard
from rlcard.agents.dqn_agent_pytorch import DQNAgent
from rlcard.agents.random_agent import RandomAgent
from rlcard.utils.utils import set_global_seed, tournament

import matplotlib.pyplot as plt


def env_load_dqn_agent_and_random_agent(agent_path=None, 
                                       trainable=False):

    # Set a global seed
    set_global_seed(0)

    # Make environment
    env = rlcard.make('nano_ofcp', config={'seed': 0})

    if trainable:
        # Load up the class of the DQN agent, and then we can populate with the weights from the checkpoint.
        agent = DQNAgent(scope='dqn',
                        action_num=env.action_num,
                        state_shape=env.state_shape,
                        mlp_layers=[64, 64],
                        device=torch.device('cpu')
                        )
    else:
        agent = DQNAgent(scope='dqn',
                        action_num=env.action_num,
                        state_shape=env.state_shape,
                        mlp_layers=[64, 64],
                        device=torch.device('cpu'), 
                        epsilon_start = 0.0,
                        epsilon_end = 0.0,
                        epsilon_decay_steps= 1
                        )

    random_agent = RandomAgent(action_num=env.action_num)

    # We have a pretrained model here. Change the path for your model.
    # check_point_path = os.path.join('rlcard/examples/models/nano_dqn_pytorch/model.pth')

    # TODO: Fix the hard coding going on here.
    if agent_path is None:
        checkpoint = torch.load('/Users/student/rlcard/models/nano_dqn_pytorch/model.pth')
    else:
        checkpoint = torch.load(agent_path)

    agent.load(checkpoint)

    # Evaluate the performance. Play with random agents.
    env.set_agents([agent, random_agent])

    return env

def play_tournament(env, evaluate_num):
    reward = tournament(env, evaluate_num)[0]
    print('Average reward against random agent: ', reward)

