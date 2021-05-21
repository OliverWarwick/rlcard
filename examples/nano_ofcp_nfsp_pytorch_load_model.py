''' An example of loading pre-trained NFSP model on Leduc Holdem
'''
import os
import torch

import rlcard
from rlcard.agents.nfsp_agent_pytorch import NFSPAgent
from rlcard.agents.random_agent import RandomAgent
from rlcard.utils.utils import set_global_seed, tournament

def load_nfsp_from_memory(check_point_path, mode):

    # Make environment
    env = rlcard.make('nano_ofcp', config={'seed': 0})

    # Set a global seed
    set_global_seed(0)

    # Load pretrained model
    nfsp_agents = []
    for i in range(env.player_num):
        agent = NFSPAgent(scope='nfsp' + str(i),
                        action_num=env.action_num,
                        state_shape=env.state_shape,
                        hidden_layers_sizes=[64, 64],
                        q_mlp_layers=[64, 64],
                        device=torch.device('cpu'), 
                        evaluate_with=mode, 
                        q_discount_factor=1.0)
        nfsp_agents.append(agent)

    # We have a pretrained model here. Change the path for your model.
    if check_point_path is None:
        check_point_path = os.path.join('./models/nano_ofcp_nfsp_result', 'model.pth')
    checkpoint = torch.load(check_point_path)
    for agent in nfsp_agents:
        agent.load(checkpoint)

    return nfsp_agents

if __name__ == '__main__':

    nfsp_agents = load_nfsp_from_memory(os.path.join('./models/nano_ofcp_nfsp_result', 'model.pth'), mode='average_policy')
    # for agent in nfsp_agents:
    #     agent._rl_agent.trainable=False
    eval_env = rlcard.make('nano_ofcp', config={'seed': 0})

    # Evaluate the performance. Play with random agents.
    evaluate_num = 100
    random_agent = RandomAgent(eval_env.action_num)
    eval_env.set_agents([nfsp_agents[0], random_agent])
    reward = tournament(eval_env, evaluate_num)[0]
    print('Average reward against random agent: ', reward)

