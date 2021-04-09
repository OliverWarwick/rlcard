import json
import os
import numpy as np
from copy import deepcopy

import rlcard
from rlcard.envs import Env
from rlcard.games.nano_ofcp import Game, ActionChoiceException
from rlcard.games.nano_ofcp.ofcp_utils import STRING_TO_RANK

DEFAULT_GAME_CONFIG = {
    'game_player_num': 2,
}

class NanoOFCPEnv(Env):

    ''' 
    Nano OFCP Environment.
    This is the outward facing interface, and will call the  underlying methods form within game. 
    This yields the functions get_payoffs, get_perfect_infomation.
    Run using the following signature: 

    env = rlcard.make('blackjack')
    env.set_agents([RandomAgent(action_num=env.action_num)])
    trajectories, payoffs = env.run()

    This makes calls from the Env super class to our version of game.


    State Encoding:
    As this is a 3 card version of OFCP suits are not important. By this I mean, if both players have triples there is no benefit to one suit over another, whereas in normal OFCP one player may have a straight flush and the other just a straight.
    As a result for each row we moniter:
        Our_Card_To_Play, Our_Front, Our_Back, Our_Discard, Oppo_Front, Oppo_Back
    we add a length 3 vector which takes the cards inside [which could be of the special int = 0 if the slot is empty] and takes the numeric value of the rank of this card e.g: J == 11, T == 10 etc...]
    e.g [[None, None, None,], [TD, TC, None], .. ] -> [0, 0, 0, 10, 10, 0, ...]
    '''

    def __init__(self, config):
        ''' Initalise the environment '''
        self.name = 'nano-ofcp'
        self.default_game_config = DEFAULT_GAME_CONFIG
        self.game = Game()
        super().__init__(config)
        self.actions = [['D', 'F', 'F'], ['D', 'F', 'B'], ['D', 'B', 'F'], ['D', 'B', 'B'], ['F', 'D', 'F'], ['F', 'D', 'B'], ['B', 'D', 'F'], ['B', 'D', 'B'], ['F', 'F', 'D'], ['F', 'B', 'D'], ['B', 'F', 'D'], ['B', 'B', 'D']]
        self.state_shape = 4 * 3 + 2 * 3 # Each row is 3 cards and we have 4 for us, with just 2 visible from the oppo.

    def _get_legal_actions(self):
        ''' Get all leagal actions

        Returns:
            encoded_action_list (list): return encoded legal action list (from str to int)
        '''
        return self.game.get_legal_actions()

    def _extract_state(self, state):
        ''' Extract the state representation from state dictionary for agent

        Note: Currently the use the hand cards and the public cards. TODO: encode the states

        Args:
            state (dict): Original state from the game

        Returns:
            observation (list): combine the player's score and dealer's observable score for observation
        '''
        extracted_state = {}
        legal_actions = [self.actions.index(a) for a in state['legal_actions']]
        extracted_state['legal_actions'] = legal_actions   

        # Processing the card information.
        extracted_state['hands'] = deepcopy(state['state'][0] + state['state'][1])

        for row in extracted_state['hands']:
            for _ in range(3-len(row)):
                row.append(None)

        
        f = lambda x: STRING_TO_RANK.get(x.rank) if x is not None else 0
        rank_vector = [[f(x) for x in row] for row in extracted_state['hands']]


        flattern_rank_vector = np.array(rank_vector).flatten()

        # We want to run through these and anywhere there is empty space we can add
        # extra zeros from the index. 
        # Obvs is what an agent can use to make a decision. This in our case is the vector embedding of our cards and those which are visable from the side of the oppo.
        extracted_state['obs'] = flattern_rank_vector

        # We pass back the raw data which is used by functions such as the human player to write out all of the cards. 
        if self.allow_raw_data:
            extracted_state['raw_obs'] = state
            extracted_state['raw_legal_actions'] = [a for a in state['legal_actions']]
        # TODO: Figure out what this does.
        if self.record_action:
            extracted_state['action_record'] = self.action_recorder

        return extracted_state

    def get_payoffs(self):
        ''' Get the payoff of a game

        Returns:
           payoffs (list): list of payoffs
        '''
        return self.game.get_payoffs()

    def _decode_action(self, action_id):

        legal_actions = self.game.get_legal_actions()
        # print("Legal actions: {}".format(legal_actions))
        if self.actions[action_id] in legal_actions:
            return self.actions[action_id]
        else:
            # Raise an exception for now we can figure this out later.
            raise ActionChoiceException("Tried to call action {} which is not possible given game state.".format(self.actions[action_id]))


    def _load_model(self):
        pass

    def get_perfect_information(self):
        ''' Get the perfect information of the current state. We can do this by using the game field and etraing all information required.
        
        Returns:
            (dict): A dictionary of all the perfect information of the current state
        '''

        state = {}
        state['player_0_hand'] = [self.game.players[0].cards_to_play, self.game.players[0].front_row, self.game.players[0].back_row, self.game.players[0].discard_pile]
        state['player_1_hand'] = [self.game.players[1].cards_to_play, self.game.players[1].front_row, self.game.players[1].back_row, self.game.players[1].discard_pile]
        state['scores'] = [self.game.players[0].score, self.game.players[1].score]
        state['current_player'] = self.game.game_pointer
        state['legal_actions'] = self.game.get_legal_actions()
        return state