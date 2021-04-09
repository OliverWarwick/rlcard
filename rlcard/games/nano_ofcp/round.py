class Nano_OFCP_Round(object):

    def __init__(self, num_players, np_random):

        self.num_players = num_players
        self.np_random = np_random
        self.game_pointer = None
        self.players_finished = [False for _ in range(num_players)]

    def start_new_round(self, game_pointer):

        self.game_pointer = game_pointer
        self.players_finished = [False for _ in range(self.num_players)]


    def proceed_round(self, players, action):
        """
        Handles the rolling forward of a round by one player taking an action.

        args:
            players: List(Player) 
            action: List(Str) - this should consist of letters like "F", "B", "D" which are the only
            three recongised characters representing front, back, and discard
            # TODO: Change these to enums at some point.
        """
        player = players[self.game_pointer]
        # print("Cards to place: {} \nAction: {}".format(player.cards_to_play, action))
        
        # Ensure that the number of cards in the players card_to_place list and then action length
        # are the same length.
        if len(player.cards_to_play) != len(action):
            raise ActionLengthException("""Length of actions not the same as the players cards to 
            play. Cards to play length: {}, 
            Action length {}""".format(len(player.cards_to_play), len(action)))

        # print("Zip of the card and placement: {}".format(zip(player.cards_to_play, action)))
        # print(len(player.cards_to_play), len(action))
        # Take the action and apply it.
        for (card, placement) in list(zip(player.cards_to_play, action)):
            # print("Card: {} Placement: {}".format(card, placement))
            if placement == "D":
                player.cards_to_play.remove(card)
                player.discard_pile.append(card)
            elif placement == "F":
                if len(player.front_row) < 3:
                    player.cards_to_play.remove(card)
                    player.front_row.append(card)
                else:
                    raise ActionChoiceException("""Attempted to add to player number {} front row.
                    This is not allowed as the player's front row already had {} 
                    cards.""".format(self.game_pointer, len(player.front_row)))
            elif placement == "B":
                if len(player.back_row) < 3:
                    player.cards_to_play.remove(card)
                    player.back_row.append(card)
                else:
                    raise ActionChoiceException("""Attempted to add to player number {} back row.
                    This is not allowed as the player's bacl row already had {} 
                    cards.""".format(self.game_pointer, len(player.back_row)))
            else:
                raise ActionChoiceException("""Attempted a move which was not recongised. Move 
                type: {}""".format(placement))
        
        
        
        # Roll the game_pointer on by one step and set the current player to be done
        self.players_finished[self.game_pointer] = True
        self.game_pointer = (self.game_pointer + 1) % self.num_players

        # Need to return the game_pointer so that game has reference to it and it does not become
        # out of sync.
        return self.game_pointer
            

    def get_legal_actions(self, players):
        
        # Use the game pointer to get the player and then remove which of the tests get failed.
        player = players[self.game_pointer]
        possibilities = [['D', 'F', 'F'], ['D', 'F', 'B'], ['D', 'B', 'F'], ['D', 'B', 'B'], ['F', 'D', 'F'], ['F', 'D', 'B'], ['B', 'D', 'F'], ['B', 'D', 'B'], ['F', 'F', 'D'], ['F', 'B', 'D'], ['B', 'F', 'D'], ['B', 'B', 'D']] 

        full_actions = []
        for poss in possibilities:
            if poss.count('F') + len(player.front_row) <= 3 and poss.count('B') + len(player.back_row) <= 3:
                full_actions.append(poss)

        return full_actions

    def is_round_over(self):
        """ Simple check if the round is done 
        
        Returns:
            (boolean): True if the current round is over
        """
        return all(self.players_finished)




class ActionLengthException(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)

class ActionChoiceException(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)