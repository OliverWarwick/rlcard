from copy import deepcopy
import numpy as np
from rlcard.games.openfacechinesepoker import Dealer, Player, Judger, NormalRound, InitalRound
from typing import List

DEFAULT_GAME_CONFIG = {
    'game_player_num': 2
}

class OpenFaceGame(object):

    def __init__(self, allow_step_back=False):
        ''' 
        Initialize the class OpenFace Game
        '''
        self.allow_step_back = allow_step_back
        self.np_random = np.random
        self.configure(game_config=DEFAULT_GAME_CONFIG)

    def configure(self, game_config):
        ''' 
        Specifiy some game specific parameters, such as player number. 
        OpenFace allows for 2 or 3 players.
        '''
        if game_config['game_player_num'] < 2 or game_config['game_player_num'] > 3:
            raise OpenFaceIncorrectNumberOfPlayersException()
        else:
            self.num_players = game_config['game_player_num']

    def init_game(self):
        ''' 
        Initialilze the game. 
        Create a dealer, and player objects. Also create a judger object to evaluate the game.

        Returns:
            state (dict): the first state of the game
            player_id (int): current player's id
        '''

        # Create a dealer and judge, and then create player objects.
        self.dealer = Dealer(self.np_random)
        self.players = [Player(i, self.np_random) for i in range(self.num_players)]
        self.judger = Judger(self.np_random)
        self.history = []
        self.round_counter = -1     # Use for the inital round, and then after that following the normal rounds.
        self.game_pointer = 0    # Used to hold which player the action is currently with. (Will beed to be configured so we can alter this whenever needed for the future (i.e: Player 1 to start rather than Player 0))
        self.score_array = dict()


        # Deal each of the players the initial 5 cards. Dealt from the dealer to each of the players.
        self.deal_to_all_players(num_cards=5)

        for i in range(self.num_players):
            self.score_array['player' + str(i)] = 0

        # Start as an inital round object, then this is switched to an one_card_round object once the inital round has been completed.
        self.round = InitalRound(num_players=self.num_players, game_pointer=self.game_pointer)

        # Counter the number of rounds of cards we have distrbuted. Starts with 0 and ends with 7.
        # TODO: STATE of the game.

        return self.get_state(self.players[self.game_pointer]), self.game_pointer


    def deal_to_all_players(self, num_cards):               

        for i in range(num_cards):
            for j in range(self.num_players):
                self.dealer.deal_card(self.players[j])

    def step(self, action) -> (List[int], int):
        ''' Get the next state

        Args:
            action (str): a specific action of blackjack. (Hit or Stand)

        Returns:/
            dict: next player's state
            int: next plater's id
        '''

        # TODO: Add in the backwards step function.
        # if self.allow_step_back:
        #     p = deepcopy(self.players[self.game_pointer])
        #     d = deepcopy(self.dealer)
        #     w = deepcopy(self.winner)
        #     self.history.append((d, p, w))

        next_state = {}

        if self.round_counter == -1:
            self.game_pointer = self.round.proceed_round(player=self.players[self.game_pointer], action=action)                
        elif self.round_counter >= 0 and self.round_counter <= 7:
            self.game_pointer = self.round.proceed_round(player=self.players[self.game_pointer], action=action)
        else:
            pass # Fix
        
        if self.round.is_over():
            if isinstance(self.round, InitalRound):
                self.round = NormalRound(num_players=self.num_players, game_pointer=self.game_pointer)
            
            # Deal a card to each player, increment the round counter and call the start_new_round method to reset values.
            self.deal_to_all_players(num_cards=1)
            self.round_counter += 1
            self.round.start_new_round(self.game_pointer)

        # FIXME: Add in a proper state function.
        state = self.get_state(self.game_pointer)

        return state, self.game_pointer

        

    def step_back(self):
        ''' Return to the previous state of the game

        Returns:
            Status (bool): check if the step back is success or not
        '''
        #while len(self.history) > 0:
        if len(self.history) > 0:
            self.dealer, self.players[self.game_pointer], self.winner = self.history.pop()
            return True
        return False


    def get_state(self, player_id) -> dict:
        ''' 
        Return player's state

        Args:
            player_id (int): player id

        Returns:
            state (dict): corresponding player's state. 
            contains the actions avalible to user, and also the state of each players hand just in terms of the normal 
        '''

        # TODO: Add in a proper state making function.

        state = {}
        state['legal_actions'] = self.round.get_legal_actions(self.players[self.game_pointer])
        for i in range(self.num_players):
            state['player'+str(i)+'_hand_front'] = self.players[i].front_row
            state['player'+str(i)+'_hand_middle'] = self.players[i].middle_row
            state['player'+str(i)+'_hand_back'] = self.players[i].back_row

        state['player'+str(self.game_pointer)+'_card_rack'] = self.players[self.game_pointer].card_rack

        return state

    def is_over(self) -> bool:
        '''
        Checks to see if all players have completed all hands, if any haven't then return false.
        '''
        for i in range(self.num_players):
            if len(self.players[i].front_row) != 3 or len(self.players[i].middle_row) != 5 or len(self.players[i].back_row) != 5:
                return False
        return True

    def get_legal_actions(self):
        return self.round.get_legal_actions(player=self.players[self.game_pointer])

    def get_payoffs(self):
        ''' Return the payoffs of the game

        Returns:
            (list): Each entry corresponds to the payoff of one player
        '''

        #TODO: Impliment this by calling the judger.

        # self.judger.judge .... 
        # return payoffs.


    


class OpenFaceIncorrectNumberOfPlayersException(Exception):
    def __init__(self):
        super().__init__("Open Face Chinese Poker allows for either 2 or 3 players because of the number of cards avalible from a deck. Please try again with a correct number of players for this game.")



# Test the game
import random 

if __name__ == "__main__":
    game = OpenFaceGame()
    # while True:
    print('New Game')
    state, game_pointer = game.init_game()
    print(game_pointer, state)
    i = 1

    while not game.is_over():
        i += 1
        
        # if i == 10:
        #     print('Step back')
        #     print(game.step_back())
        #     game_pointer = game.get_player_id()
        #     print(game_pointer)
        #     legal_actions = game.get_legal_actions()

        legal_actions = game.get_legal_actions()
        action = random.choice(legal_actions)
        # print(game_pointer, action, legal_actions)
        state, game_pointer = game.step(action)
        print(game_pointer, state)

    #    print(game.get_payoffs())
