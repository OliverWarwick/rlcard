''' Play off the best dqn agent vs a human player '''

''' 
Termial based GUI for playing against an agent. Should be useful for debugging against agents and testing the enumlator works.
'''

import rlcard
import torch
from rlcard.agents import DQNAgentPytorch as DQNAgent
from rlcard.agents import NanoOFCPHumanAgent as HumanAgent
from rlcard.utils.utils import print_card
from nano_ofcp_dqn_pytorch_load_model import env_load_dqn_agent_and_random_agent

CARDS_TO_PLAY = 0
FRONT_ROW = 1 
BACK_ROW = 2
DISCARD_PILE = 3
OPPO_FRONT_ROW = 0
OPPO_BACK_ROW = 1

agent_path = 'models/nano_dqn_pytorch/best_model_updated_128.pth'

# Make environment and enable human mode
# Set 'record_action' to True because we need it to print results
player_num = 2
env = rlcard.make('nano_ofcp', config={'record_action': True, 'game_player_num': player_num})

human_agent = HumanAgent(env.action_num)
dqn_agent = DQNAgent(scope='dqn',
                        action_num=env.action_num,
                        state_shape=env.state_shape,
                        mlp_layers=[128, 128],
                        device=torch.device('cpu'), 
                        epsilon_start = 0.0,
                        epsilon_end = 0.0,
                        epsilon_decay_steps= 1
                        )
checkpoint = torch.load(agent_path)
dqn_agent.load(checkpoint)


env.set_agents([human_agent, dqn_agent])

running_totals = [0, 0]
num_round = 0

print(">> Nano OFCP human agent")

while True:
    print(">> Start a new game")
    num_round += 1

    trajectories, payoffs = env.run(is_training=False)
    # If the human does not take the final action, we need to
    # print other players action
   
    if len(trajectories[0]) != 0:
        final_state = []
        action_record = []
        state = []
        _action_list = []

        for i in range(player_num):
            final_state.append(trajectories[i][-1][-2])
            state.append(final_state[i]['raw_obs'])

        action_record.append(final_state[i]['action_record'])
        for i in range(1, len(action_record) + 1):
            _action_list.insert(0, action_record[-i])

        for pair in _action_list[0]:
            print('>> Player', pair[0], 'chooses', pair[1])
    
    # print("State i of state: {}".format(state[i]['state']))
    # print(state[i])

    # Get the ending arrangement.
    my_cards = state[i]['state'][0]
    oppo_cards = state[i]['state'][1] # Unfortunate naming - maybe change in future.
   
    print('\n===============   MY HAND  ===============\n')
    print('===============   Front Row       ===============')
    print_card(my_cards[FRONT_ROW])
    print('===============   Back Row        ===============')
    print_card(my_cards[BACK_ROW])

    print('\n===============   OPPO HAND   ===============\n')
    print('===============   Front Row       ===============')
    print_card(oppo_cards[OPPO_FRONT_ROW])
    print('===============   Back Row        ===============')
    print_card(oppo_cards[OPPO_BACK_ROW])
    print('===============     Result     ===============')

    print(payoffs)

    # In OFCP there are only one payoff which comes at the end, so payoff will only have 1 value..
    for i in range(player_num):
        # Add to the running totals.
        running_totals[i] = running_totals[i] + payoffs[i]
        print("Payoff for player {}: {}".format(i, payoffs[i]))
        print('')

    response = input("Press q to quit, or any key to continue: ")
    if response.startswith('q'):
        break

# Output the final scores.
print('\n===============   FINAL SCORES  ===============\n')
for i in range(player_num):
        # Add to the running totals.
        print("Player {}".format(i))
        print("Final score: {}".format(payoffs[i]))
        print("Avg number of points per round: {}".format(payoffs[i] / num_round))
        print('')
