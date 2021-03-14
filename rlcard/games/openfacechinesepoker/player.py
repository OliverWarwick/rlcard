class OpenFacePlayer(object):

    def __init__(self, player_id, np_random):
        ''' Initialize a OpenFace player class

        Args:
            player_id (int): id for the player
            np_random (numpy random object): used for sampling at random.
        '''
        self.np_random = np_random
        self.player_id = player_id

        self.front_row = []
        self.middle_row = []
        self.back_row = []

        self.card_rack = []

        self.status = 'alive'
        self.score = 0

    def get_player_id(self):
        ''' 
        Return player's id
        '''
        return self.player_id

    def place_card(self, row_to_place):
        '''
        Remove the card from the card rank and place into a row.
        Throws an Exception should there be any problem in placement because of the row being full, or because the row was incorrect.
        TODO: Make enums for the back middle and front rows. 

        Args: 
            row_to_place (str) : front, middle or back.
        '''
        if len(self.card_rack) == 0:
            raise EmptyCardRackException("Tried to place a card which does not exist.")

        if row_to_place == "front" and len(self.front_row) < 3:
            self.front_row.append(self.card_rack.pop())
        elif row_to_place == "middle" and len(self.middle_row) < 5:
            self.middle_row.append(self.card_rack.pop())
        elif row_to_place == "back" and len(self.back_row) < 5:
            self.back_row.append(self.card_rack.pop())
        else:
            raise RowPlacementException("Incorrect row for placement either in name or because the {} row was full".format(row_to_place))


class EmptyCardRackException(Exception):
    def __init__(self, message):
        super().__init__(message)
    
class RowPlacementException(Exception):
    def __init__(self, message):
        super().__init__(message)



if __name__ == "__main__":
    print("Hello World")