''' File for useful functions which are used in OFCP, such as comparision methods. 
These are likely to be shared across the different games so seems easier to place them in here. 

This follows the same format as the utils.py file within rlcard/games/limit_holdem. However I've
had to modify this because hold'em features 7 cards (2 pocket + 5 community) which would break
several of the functions. 

For now we are only playing with up to triples, so it makes sense to keep the code simple, but some
time should be investated to make this good at some point.'''

from collections import Counter
from rlcard.utils.utils import init_mini_deck

# Useful maps for turning ranks -> ints and back.
RANK_TO_STRING = {2: "2", 3: "3", 4: "4", 5: "5", 6: "6",
                               7: "7", 8: "8", 9: "9", 10: "T", 11: "J", 12: "Q", 13: "K", 14: "A"}
STRING_TO_RANK = {v:k for k, v in RANK_TO_STRING.items()}
# hand_func_list is in top to bottom order so that we can quite once we onbtain a ranking.        


class Row:

    """
    Helper function mainly for comparing rows.

    """

    def __init__(self, cards):

        self.cards = cards
        self.suit_freq = None
        self.rank_freq = None

        self.poker_hand_func_list = [
            (self._has_royal_flush, "Royal Flush"), 
            (self._has_staight_flush, "Straight Flush"), 
            (self._has_four_of_kind, "Four of a Kind"),
            (self._has_full_house, "Full House"),
            (self._has_flush, "Flush"), 
            (self._has_straight, "Straight"),
            (self._has_three_of_kind, "Three of a Kind"),
            (self._has_two_pair, "Two Pair"),
            (self._has_pair, "One Pair"), 
            (self._has_high_card, 'High Card')  
        ]

    def __lt__(self, other):

        index_left, _, rank_left = self.evaluate_row()
        index_right, _, rank_right = other.evaluate_row()
 
        if index_left != index_right:
            # Indexing from 0 -> 9, with 0 being the best so in the case of the self smaller than 
            # other we want index_left > index_right.
    
            return index_left > index_right
        else:
            # They must have the same hand, so we can check rank perform a more through check.
            numeric_rank_left = STRING_TO_RANK.get(rank_left)
            numeric_rank_right = STRING_TO_RANK.get(rank_right)

            if numeric_rank_left != numeric_rank_right:
                # These are in normal order we can compare.
                return numeric_rank_left < numeric_rank_right
            else:
                # Now need to check these are the same the whole way down.
                # At this point they must have the same rank, and same best cards, so we can just 
                # loop through the rank freq and see which has the better results as this is pre
                # sorted into the correct fashion.
                merged_rank_freq = list(zip(self.rank_freq, other.rank_freq))
                for left, right in merged_rank_freq:
                    numeric_left = STRING_TO_RANK.get(left[0])
                    numeric_right = STRING_TO_RANK.get(right[0])
                    if numeric_left != numeric_right:
                        return numeric_left < numeric_right
                return False

    def __eq__(self, other):

        index_left, _, rank_left = self.evaluate_row()
        index_right, _, rank_right = other.evaluate_row()
        # print("Left index: {}, Right index: {}, Left rank: {}, Right rank: {}".format(index_left, index_right, rank_left, rank_right))
        if index_left == index_right and rank_left == rank_right:
            # Check whether the ranks are the same because we know they are the same type of hand,
            # we also know that the hands are ordered by the most commonly occuring card, then 
            # by the highest valued card of that frqeuece.
            merged_rank_freq = list(zip(self.rank_freq, other.rank_freq))
            for left, right in merged_rank_freq:
                if left[0] != right[0]:
                    return False
            return True
        else:
            return False

    def __str__(self):
        return str(self.cards)

    def __repr__(self):
        return self.__str__()
    
    def evaluate_row(self):

        """ 
        Returns the poker hand and relavent highest card of a row from OFCP. 
        
        Raises:
            EmptyRowException: Should it be called on an empty row .

        Returns:
            index: (int): Index in the list of poker hands where this occurs. 0 = Royal Flsuh, 9 = High Card.
            hand (str): poker hand e.g: High Card, One Pair    
            rank: (str): highest ranked card which was relevant e.g: T, J.    
        """

        # First we call to get the rank and suit frequency.
        self._get_rank_freq()
        self._get_suit_freq()

        # Starting from the best hand and working down we can look through the list of possible
        # checks and if a hand returns true then can return as well. 

        for index, (func, hand) in enumerate(self.poker_hand_func_list):
            hand_exists, rank = func()
            if hand_exists:
                return index, hand, rank   
        raise EmptyRowException("Tried to evaluate an empty row.")
          
    ''' 
    Helper functions for setting up the Counter objects. Specifically, we order first by number
    of occurances, and then by the rank in the rank case. For example [JD, JS, TD, TS] -> [('J', 2),
    ('T', 2)] which makes it easier to then deal with getting pairs out. 
    This also works for identifying two pairs etc, because we sort by the freq, then by the 
    numeric value, which means even if the non-paired card is higher than the paired cards
    we 
    OW - Is this quicker, test.
    '''
    def _get_suit_freq(self):
        self.suit_freq = sorted(Counter(card.suit for card in self.cards).items(), 
        key=lambda x: (x[1]), reverse=True)

    def _get_rank_freq(self):
        self.rank_freq = sorted(Counter(card.rank for card in self.cards).items(), 
        key=lambda x: (x[1], STRING_TO_RANK.get(x[0])), reverse=True)

    ''' 
    Helper functions for determining which poker hand has been formed.
    Two Pre-Conditions:
        self.rank_freq and self.suit_freq must have at least one element.
        suit_freq and rank_freq are up to date. 
    '''
    def _has_high_card(self):
        rank, freq = self.rank_freq[0]
        return (True, rank)

    # These could be abstracted out to save some code but I find them more readable / interpretable in this way.
    def _has_pair(self):
        rank, freq = self.rank_freq[0]
        if freq >= 2:
            return (True, rank) 
        return (False, None)
    
    def _has_two_pair(self):
        if len(self.rank_freq) > 1:
            rank_pair_1, freq_pair_1 = self.rank_freq[0]
            rank_pair_2, freq_pair_1 = self.rank_freq[1]
            if freq_pair_1 == 2 and freq_pair_1 == 2:
                return (True, rank_pair_1)
        return (False, None)

    def _has_three_of_kind(self):
        rank, freq = self.rank_freq[0]
        if freq >= 3:
            return (True, rank) 
        return (False, None)

    def _has_straight(self):
        # We can check if every frequency is 1, and if so whether the first and last (in order)
        # are 4 apart (3, 4, 5, 6, 7) as this is an eq condition under our rank and suit ordering
        if len(self.cards) != 5:
            return (False, None)
        does_have_straight = True
        for (rank, freq) in self.rank_freq:
            if freq != 1: 
                does_have_straight = False

        if does_have_straight and STRING_TO_RANK.get(self.rank_freq[0][0]) - STRING_TO_RANK.get(self.rank_freq[-1][0]) == 4:
            # Return the rank of the best  card.
            return (True, self.rank_freq[0][0])
        return (False, None)
    
    def _has_flush(self):
        _, freq = self.suit_freq[0]
        if freq == 5:
            # Need to run to find the top card. 
            rank, freq = self.rank_freq[0]
            return (True, rank)
        return (False, None)

    def _has_four_of_kind(self):
        rank, freq = self.rank_freq[0]
        if freq >= 4:
            return (True, rank) 
        return (False, None)
    
    def _has_full_house(self):
        if len(self.rank_freq) > 1:
            rank_trip, freq_trip = self.rank_freq[0]
            rank_pair, freq_pair = self.rank_freq[1]
            if freq_trip == 3 and freq_pair == 2:
                return (True, rank_trip)
        return (False, None)
    
    def _has_staight_flush(self):
        has_straight, rank = self._has_straight()
        has_flush, _ = self._has_flush()
        if has_straight and has_flush:
            return (True, rank)
        return (False, None)

    def _has_royal_flush(self):
        has_straight, rank = self._has_straight()
        has_flush, _ = self._has_flush()
        # Need to check we have a straight flush, and that the rank so the highest card 
        # which will be returned from _has_straight is an ACE = 14.
        if has_straight and has_flush and rank == "A":
            return (True, rank)
        return (False, None)


        

class EmptyRowException(Exception):
    """
    Raised anytime we try and evaluate or compare an empty row.
    """
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)
    

    

if __name__ == '__main__':
    deck = init_mini_deck()
    row1 = Row([deck[5], deck[5], deck[6]])
    row2 = Row([deck[10], deck[10], deck[11]])
    print(row1)
    print(row2)
    row1._get_rank_freq()
    row2._get_suit_freq()
    print(row1.evaluate_row())
    print(row2.__eq__(row1))