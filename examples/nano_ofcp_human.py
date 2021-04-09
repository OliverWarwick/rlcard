''' 
Termial based GUI for playing against an agent. Should be useful for debugging against agents and testing the enumlator works.
'''

import rlcard
from rlcard.agents import RandomAgent as RandomAgent
from rlcard.agents import NanoOFCPHumanAgent as HumanAgent
from rlcard.utils.utils import print_card

CARDS_TO_PLAY = 0
FRONT_ROW = 1 
BACK_ROW = 2
DISCARD_PILE = 3
OPPO_FRONT_ROW = 0
OPPO_BACK_ROW = 1

# Make environment and enable human mode
# Set 'record_action' to True because we need it to print results
player_num = 2
env = rlcard.make('nano_ofcp', config={'record_action': True, 'game_player_num': player_num})
human_agent = HumanAgent(env.action_num)
random_agent = RandomAgent(env.action_num)
env.set_agents([human_agent, random_agent])

print(">> Nano OFCP human agent")

while True:
    print(">> Start a new game")

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
        print("Payoff for player {}: {}".format(i, payoffs[i]))
        print('')

    input("Press any key to continue...")
