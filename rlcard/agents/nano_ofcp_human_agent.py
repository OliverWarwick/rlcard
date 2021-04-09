from rlcard.utils.utils import print_card

# Copied from HumanAgent within blackjack_human_agent.py within agents directory.

# Helpful names for the rest of the 
CARDS_TO_PLAY = 0
FRONT_ROW = 1 
BACK_ROW = 2
DISCARD_PILE = 3
OPPO_FRONT_ROW = 0
OPPO_BACK_ROW = 1

class HumanAgent:
    ''' A human agent for Blackjack. It can be used to play alone for understand how the blackjack code runs
    '''

    def __init__(self, action_num):
        ''' Initilize the human agent

        Args:
            action_num (int): the size of the output action space
        '''
        self.use_raw = True
        self.action_num = action_num

    @staticmethod
    def step(state):
        ''' Human agent will display the state and make decisions through interfaces

        Args:
            state (dict): A dictionary that represents the current state

        Returns:
            action (int): The action decided by human
        '''
        _print_state(state['raw_obs'], state['raw_legal_actions'], state['action_record'])
        input_string = input('>> You choose action (three letters [D, F, B]): ')
        # Split this up.
        action_chars = [char for char in input_string]
        print(action_chars)
        print(state['raw_legal_actions'])
        while action_chars not in state['raw_legal_actions']:
            print('Action illegel...')
            input_string = input('>> You choose action (three letters [D, F, B]): ')
            action_chars = [char for char in input_string]
        action = state['raw_legal_actions'].index(action_chars)
        return state['raw_legal_actions'][action]

    def eval_step(self, state):
        ''' Predict the action given the current state for evaluation. The same to step here.

        Args:
            state (numpy.array): an numpy array that represents the current state

        Returns:
            action (int): the action predicted (randomly chosen) by the random agent
            probs (list): The list of action probabilities
        '''
        return self.step(state), []

def _print_state(state, raw_legal_actions, action_record):
    ''' 
    Print out the state

    Args:
        state (dict): A dictionary of the raw state
        action_record (list): A list of the each player's historical actions
    '''
    _action_list = []
    for i in range(1, len(action_record)+1):
        _action_list.insert(0, action_record[-i])
    # Don't require a summery of moves in OFCP, but may be useful later.
    # for pair in _action_list:
    #     print('>> Player', pair[0], 'chooses', pair[1])


    num_player = 2
    my_cards = state['state'][0]
    oppo_cards = state['state'][1] # Unfortunate naming - maybe change in future.
   

    print('\n===============   MY HAND  ===============\n')
    print('===============   Card to Play       ===============')
    print_card(my_cards[CARDS_TO_PLAY])
    print('===============   Front Row       ===============')
    print_card(my_cards[FRONT_ROW])
    print('===============   Back Row        ===============')
    print_card(my_cards[BACK_ROW])
    print('\n===============   OPPO HAND   ===============\n')
    print('===============   Front Row       ===============')
    print_card(oppo_cards[OPPO_FRONT_ROW])
    print('===============   Back Row        ===============')
    print_card(oppo_cards[OPPO_BACK_ROW])
        
    print(raw_legal_actions)
    print('\n=========== Actions You Can Choose ===========')
    print(', '.join([str(index) + ': ' + str(action) for index, action in enumerate(raw_legal_actions)]))
    print('')