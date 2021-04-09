class Nano_OFCP_Player(object):

    """
    
    Fields:
    player_id (int): Either 1 or 2.
    front_row (list[Card]): List of card objects in the front row. Up to three cards.
    back_row (list[Card]): List of card objects in the back row. Up to three cards.
    cards_to_player (list[Card]): List of Card objects which the player must place down into the front or back hand by
                                  discarding one and playing the other two.
    status: (str): alive if not bust o/w dead.
    score: (int): score obtained so far in the game.
    """

    def __init__(self, player_id, np_random):
        ''' Initialize a Blackjack player class

        Args:
            player_id (int): id for the player
        '''
        self.np_random = np_random
        self.player_id = player_id
        self.front_row = []
        self.back_row = []
        self.cards_to_play = []
        self.discard_pile = []
        self.status = 'alive'
        self.score = 0

    def get_player_id(self):
        ''' Return player's id
        '''
        return self.player_id

