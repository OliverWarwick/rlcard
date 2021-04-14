from rlcard.games.nano_ofcp import Dealer
from rlcard.games.nano_ofcp import Judger
from rlcard.games.nano_ofcp import Player
from rlcard.games.nano_ofcp import Round

from copy import deepcopy
import numpy as np
import random

class Nano_OFCP_Game(object):

    def __init__(self, allow_step_back=False, num_players=2):
        ''' Initialize the class limitholdem Game
        '''
        self.allow_step_back = allow_step_back
        self.np_random = np.random.RandomState()
        self.num_players = 2
        self.action_num = 12
        self.player_to_lead = None

    def configure(self, game_config):
        ''' Specifiy some game specific parameters, such as player number
        '''
        self.player_num = game_config['game_player_num']

    def init_game(self):
        self.dealer = Dealer(self.np_random)
        self.players = [Player(0, self.np_random), Player(1, self.np_random)]
        self.judger = Judger(self.np_random)

        # Player to play first should alternate between the players in order to not give
        # an advantaged to over the other.
        if self.player_to_lead is None:
            self.player_to_lead = random.randint(0,2)
        else:
            self.player_to_lead = (self.player_to_lead + 1) % self.num_players

        self.game_pointer = self.player_to_lead
        self.history = []
        self.round = Round(num_players=self.num_players,
                           np_random=self.np_random)
        self.round.start_new_round(self.game_pointer)

        # Deal the initial cards, and add one to the dealer count.
        for i in range(self.num_players):
            self.dealer.deal_triple(self.players[i])

        self.dealer.deal_counter += 1
        state = self.get_state(self.game_pointer)

        return state, self.game_pointer

    def step(self, action):

        """[summary]
        """

        if self.allow_step_back:
            g_p = deepcopy(self.game_pointer)
            l_p = deepcopy(self.player_to_lead)
            d_c = deepcopy(self.dealer.deal_counter)
            j = deepcopy(self.judger)
            d = deepcopy(self.dealer)
            p = deepcopy(self.players)
            r = deepcopy(self.round)
            self.history.append((p, r, d, j, g_p, l_p, d_c))

        # Step the round forward.
        self.game_pointer = self.round.proceed_round(self.players, action)

        # If the round is over then we can deal the next set of cards, if not the last turn.
        if self.round.is_round_over():
            if self.dealer.deal_counter <= 2:
                for i in range(self.num_players):
                    self.dealer.deal_triple(self.players[i])
                
                # Add one to the deal_counter and set up a new round.
                self.dealer.deal_counter += 1
                self.round.start_new_round(self.game_pointer)

        
        state = self.get_state(self.game_pointer)
        return state, self.game_pointer

    def step_back(self):
        ''' Return to the previous state of the game

        Returns:
            (bool): True if the game steps back successfully
        '''
        if len(self.history) > 0:
            self.players, self.round, self.dealer, self.judger, self.game_pointer, self.player_to_lead, self.deal_counter = self.history.pop() 
            return True
        return False

    def get_player_num(self):
        ''' Return the number of players in Limit Texas Hold'em

        Returns:
            (int): The number of players in the game
        '''
        return self.num_players

    @staticmethod
    def get_action_num():
        ''' Return the number of applicable actions

        Returns:
            (int): The number of actions. There are 4 actions (call, raise, check and fold)
        '''
        return 12

    def get_player_id(self):
        ''' Return the current player's id

        Returns:
            (int): current player's id
        '''
        return self.game_pointer

    def is_over(self):
        """
        Returns whether the game is over or not.

        Returns:
            [bool]: whether the game is over in which can we get the payoffs.
        """
        return self.round.is_round_over() and self.dealer.deal_counter > 2
            
    def get_payoffs(self):
        """[summary]

        Returns:
            tuple (int, int): scores for each of the players.
        """

        # Can make a call judge game to get the results form scoring, currently comes back a tuple.
        return self.judger.judge_game(self)

    def get_legal_actions(self):
        ''' Return the legal actions for current player

        Returns:
            (list): A list of legal actions
        '''
        return self.round.get_legal_actions(self.players)


    def get_state(self, player):
        ''' Return player's state

        Args:
            player_id (int): player id

        Returns:
            (dict): The state of the player
        '''
        next_state = {} 
        for i in range(self.num_players):
            next_state['player' + str(i) + 'hand'] = [self.players[player].cards_to_play, self.players[player].front_row, self.players[player].back_row, self.players[player].discard_pile]
        next_state['legal_actions'] = self.get_legal_actions()
        next_state['score'] = self.get_payoffs()

        # Prepare state which is what will be used to create the vector embedding as this is the viewable information.
        oppo_game_pointer = (self.game_pointer + 1) % self.num_players
        next_state['state'] = ([self.players[self.game_pointer].cards_to_play, self.players[self.game_pointer].front_row, self.players[self.game_pointer].back_row, self.players[self.game_pointer].discard_pile], [self.players[oppo_game_pointer].front_row, self.players[oppo_game_pointer].back_row])
       
        return next_state


# TODO: Add in all the other functions from game.py in the limit holden case. Seem to be some access modifers.

# Test the game

if __name__ == "__main__":
    game = Nano_OFCP_Game()
    # while True:
    print('Starting Game')
    state, game_pointer = game.init_game()
    print(game_pointer, state)
    i = 1
    while not game.is_over():
        i += 1
        print("START OF TURN {} for player {}".format(i // 2, game_pointer))

        print("Player {} move \nCurrent cards to play: {} \nCurrent front row: {} \nCurrent back row: {}".format(game_pointer, game.players[game_pointer].cards_to_play, game.players[game_pointer].front_row, game.players[game_pointer].back_row))
        legal_actions = game.get_legal_actions()
        action = legal_actions[np.random.choice(range(0, len(legal_actions)))]
        print(action)

        print("\nPre Step")
        print("Game Pointer: {}. \nAction: {} \nLegal Actions: {}\n".format(game_pointer, action, legal_actions))
        state, game_pointer = game.step(action)
        print("Post Step\n")
        print("Game Pointer: {}. \nState: {} \n".format(game_pointer, state))
        print("END OF TURN {} for player {}".format(i // 2, game_pointer))

    print(game.get_payoffs())