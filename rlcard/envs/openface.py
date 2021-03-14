# Wrapper class for enact a game of open face.



import rlcard
from rlcard.envs import Env
from rlcard.games.openfacechinesepoker import Game
import random
import numpy as np

DEFAULT_GAME_CONFIG = {
        'game_player_num': 2,
}

# There is a very similar one of these in limitholdem/util.py but this translates 2:2 and so on, which means the index 1 is never used.
EMBEDDING_RANK_TO_INDEX = {'A': 0, '2': 1, '3': 2, '4': 3, '5': 4, '6': 5, '7': 6, '8': 7, '9': 8, 'T': 9, 'J': 10, 'Q': 11, 'K': 12}

class OpenFaceEnv(Env):

    # Functions to include:
    # __init__
    # _get_legal_actions()
    # _extract_state()
    # get_payoffs()
    # _decode_action()
    # _load_model()
    # get_perfect_information()


    def __init__(self, config):

        self.SUIT_LOOKUP = "SCDH"

        self.name = 'open-face'
        self.default_game_config = DEFAULT_GAME_CONFIG
        self.game = Game()
        super().__init__(config)
        self.actions = self._get_legal_actions()

        self.state_space = None         # TODO: Pick a method for dealing with the shape space.

    
    def _get_legal_actions(self):
        ''' Get all legal actions in the game, this will depend on which round and who the player is.

        Returns:
            List(Str): This is a list with each move avalible, when this is the inital move the list consists of tuples of the moves e.g: ('Front', 'Back', ...) but when this is a normal hand the list consists of strings representing moves e.g: ['front', middle']
        '''
        return self.game.get_legal_actions()

    

    def _extract_state(self, state):
        ''' Extract state from the agent in a dictionary 
        This should take the underlying game representation of state (raw from dictionary) and translate this into a single vector embedding.
        
        Returns:
            dict: ....
        '''

        extracted_state = {}

        legal_actions = state['legal_actions']
        # Need to then encode the actions as one-hot vectors in order to use for the output of the network.
        # TODO: Will need to figure out how to make this consistent in the future, so same number of actions in each state rather than very large at the start, and 

        # Idea is to take the players vector, and for each generate a vector which embeds the infomation.
        observations = np.zeros(self.game.num_players * (13 * 13 + 4 * (4 + 6 + 6)) + 5 * 13)               
        # Current need 13 dim for each card, then additionally 4 sets of (4 front hand + 6 for both back and middle) for recording the number of each suit.
        
        index = 0
        for i in range(self.game.num_players):

            # Front Row - Values
            for card in state['player'+str(i)+'_hand_front']:
                rank = EMBEDDING_RANK_TO_INDEX[card.rank]
                observations[index+ rank] = 1
                index += 13                             # 13 card ranks, so jump to the next vector.
            # Middle Row - Values
            for card in state['player'+str(i)+'_hand_middle']:
                rank = EMBEDDING_RANK_TO_INDEX[card.rank]
                observations[index+ rank] = 1
                index += 13
            # Back Row - Values
            for card in state['player'+str(i)+'_hand_back']:
                rank = EMBEDDING_RANK_TO_INDEX[card.rank]
                observations[index+ rank] = 1
                index += 13

            # Front Row - Suits
            card_string = ''.join(state['player'+str(i)+'_hand_front'])
            for suit in self.SUIT_LOOKUP:
                suit_count = card_string.count(suit)
                observations[index + suit_count] = 1
                index += 4
            # Middle Row - Suits
            card_string = ''.join(state['player'+str(i)+'_hand_middle'])
            for suit in self.SUIT_LOOKUP:
                suit_count = card_string.count(suit)
                observations[index + suit_count] = 1
                index += 6
            # Back Row - Suits
            card_string = ''.join(state['player'+str(i)+'_hand_back'])
            for suit in self.SUIT_LOOKUP:
                suit_count = card_string.count(suit)
                observations[index + suit_count] = 1
                index += 6
        
        # TODO: Need to add on the information encoded in the card rack of the player. Also need to test this gives the right info about the player and not the other persons card.
        # Could be up to 5 cards, but most of the time will be just 1. Think about how to fix this.
        for card in state['plauer'+self.game.game_pointer+'_card_rack']:
            rank = EMBEDDING_RANK_TO_INDEX[card.rank]
            observations[index+ rank] = 1
            index += 13 

        extracted_state['obs'] = observations
        extracted_state['legal_actions'] = legal_actions

        # TODO: May be a really nice way to do this - have a think on it.
        # In the future may need to add past actions and so on. 

        if self.allow_raw_data:
            extracted_state['raw_obs'] = state
            extracted_state['raw_legal_actions'] = [a for a in state['legal_actions']]
        if self.record_action:
            extracted_state['action_record'] = self.action_recorder

        return extracted_state



    def get_payoffs(self):
        ''' Get the payoffs from the game
        
        Returns: 
            payoffs (list): list of payoffs.
        '''
        return self.game.get_payoffs()          # FIXME: Need to impliment in the game class.

    
    def _decode_action(self, action_id):
        ''' Decode the action for applying to the game. Appears to be that if the move is not possible we should complete a move which is allowed rather than allowing the game to crash '''
        # Should this be the case in ours? Should we instead just terminate the trajectory to ensure we are generating true results.
        '''
        Args:
            action id (int): action id

        Returns:
            action (str): action for the game
        '''

        legal_actions = self.game.get_legal_actions()
        if self.actions[action_id] in legal_actions:
            return self.actions[action_id]
        else:
            raise BadActionException("Attempted to play action {} - not possible given the current game state.".format(self.actions[action_id]))


    def _load_model(self):
        ''' Load pretrain/rule model

        Returns: 
            model (Model): A model object.
        '''
        #TODO: Impliment - need to build a pretrained / rule based model to start with.

    def get_perfect_information(self):
        ''' Returns all of the information about the game, from the perfect observer. May not be useful in this game as perfect information state = normal state. 
        '''
        #TODO: Impliment.



class BadActionException(Exception):
    def __init__(self, message):
        super().__init__(message)





    


