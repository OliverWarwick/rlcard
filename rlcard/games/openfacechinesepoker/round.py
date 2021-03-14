from typing import List
import itertools

class OpenFaceInitalRound():

    def __init__(self, num_players, game_pointer):

        ''' 
        This is when both players have 5 cards, and must distrubute them between the 3 rows of their hands.
        '''
        self.num_players = num_players
        self.start_new_round(game_pointer=game_pointer)


    def start_new_round(self, game_pointer):
        
        self.game_pointer = game_pointer    # To hold the current player who is about to act.
        self.players_completed = 0          # Moniter the number of players who have played so far.
    
    def proceed_round(self, player, action) -> int:

        '''
        Enact one action providing it is valid, and then move the game pointer to the next player.

        Args:
            players OpenFacePlayer: list of players in the game.
            action str: "front", "middle" or "back".
        '''

        if not self.check_legal_action(action=action):
            raise InvalidActionException("Action {} was not possible.".format(action))

        # Place the card into the row which is stated in the action.
        for ac in action:
            player.place_card(ac)

        self.game_pointer = (self.game_pointer + 1) % self.num_players      # Get the next player, which could be player 0.
        self.players_completed += 1

        return self.game_pointer 

    
    def is_over(self):
        '''Return whether we have readed the end of the round or not.'''
        return self.players_completed >= self.num_players

    def check_legal_action(self, action: List[str]) -> bool:
        '''
        Quick check to ensure that no more than 3 cards will be placed in the front hand. Any other combination is possible.

        Args:
            action: List[str] - this is the allocation of cards in order to hands. e.g: ['front', 'back', 'back', 'middle', 'middle']
        '''
        return (len(action) != 5) or (action.count('front')) <= 3

    def get_legal_actions(self, player) -> List:

        '''
        Return a list of hands which are possible to place cards into
        '''
        return [ac for ac in itertools.product(['front', 'middle', 'back'], repeat=5) if ac.count('front') <= 3]

    def get_player_num(self):
        return self.num_players
    @staticmethod
    def get_action_num():
        return 232              # This is 3^5 minus the number of position which are invalid because of more than 3 front row cards.



class OpenFaceNormalRound(OpenFaceInitalRound):

    def __init__(self, num_players, game_pointer):
        '''
        Initaliser, but we reset the values at the start of each round, so to save creating new objects, call the start_new_round method to set the values
        '''
        super().__init__(num_players, game_pointer)
    
    def proceed_round(self, player, action) -> int:

        '''
        Enact one action providing it is valid, and then move the game pointer to the next player.

        Args:
            players OpenFacePlayer: list of players in the game.
            action str: "front", "middle" or "back".
        '''

        if action not in self.get_legal_actions(player):
            raise InvalidActionException("Action {} was not possible.".format(action))

        # Place the card into the row which is stated in the action.
        player.place_card(action)

        self.game_pointer = (self.game_pointer + 1) % self.num_players      # Get the next player, which could be player 0.
        self.players_completed += 1

        return self.game_pointer 

    
    def is_over(self):
        '''Return whether we have readed the end of the round or not.'''
        return self.players_completed >= self.num_players


    def get_legal_actions(self, player) -> List:

        '''
        Return a list of hands which are possible to place cards into
        '''

        full_actions = ['front', 'middle', 'back']

        if len(player.front_row) >= 3:
            full_actions.remove('front')
        if len(player.middle_row) >= 5:
            full_actions.remove('middle')
        if len(player.back_row) >= 5:
            full_actions.remove('back')

        return full_actions

    def get_player_num(self):
        return self.num_players
    @staticmethod
    def get_action_num():
        return 232
            
class InvalidActionException(Exception):
    def __init__(self, message):
        super().__init__(message)

