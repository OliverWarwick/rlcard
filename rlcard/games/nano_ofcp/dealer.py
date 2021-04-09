from rlcard.utils import init_mini_deck
from rlcard.games.nano_ofcp.ofcp_utils import STRING_TO_RANK
import numpy as np

class Nano_OFCP_Dealer(object):

    """
    OFCP Dealer holds no cards and is responsible only for dealing the cards to players.
    """

    def __init__(self, np_random):
        ''' Initialize a Blackjack dealer class
        '''
        self.np_random = np_random
        self.deck = init_mini_deck()
        self.shuffle()
        self.deal_counter = 0
        self.status = 'alive'
        self.score = 0

    def shuffle(self):
        ''' Shuffle the deck
        '''
        shuffle_deck = np.array(self.deck)
        self.np_random.shuffle(shuffle_deck)
        self.deck = list(shuffle_deck)

    def deal_card(self, player):
        ''' Distribute one card to the player

        Args:
            player_id (int): the target player's id
        '''
        card = self.deck.pop()
        player.hand.append(card)

    def deal_triple(self, player):

        """ Used to deal a triple of cards to a player as is done in each hand.

        Args:
            player_id (int): the target player's id.
        """
        cards = sorted([self.deck.pop() for _ in range(3)], key=lambda card: (STRING_TO_RANK.get(card.rank), card.suit), reverse=True)
        player.cards_to_play = cards
