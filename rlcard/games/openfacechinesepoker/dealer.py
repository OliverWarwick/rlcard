from rlcard.utils import init_standard_deck
import numpy as np

class OpenFaceDealer(object):

    '''
    Dealer object is used purely to distribute cards, they do not a hand and play no part in the game
    '''

    def __init__(self, np_random):
        ''' 
        Initialize a OpenFace dealer class
        '''
        self.np_random = np_random
        self.deck = init_standard_deck()
        self.shuffle()

    def shuffle(self):
        ''' 
        Shuffle the deck
        '''
        shuffle_deck = np.array(self.deck)
        self.np_random.shuffle(shuffle_deck)
        self.deck = list(shuffle_deck)

    def deal_card(self, player):
        ''' 
        Distribute one card to the player

        Args:
            player Player: OpenFacePlayer object.
        '''
        if len(self.deck) > 0:
            player.card_rack.append(self.deck.pop())
        else: 
            raise EmptyDeckException("Tried to remove a card from the empty deck")
        
    
    class EmptyDeckException():
        pass
