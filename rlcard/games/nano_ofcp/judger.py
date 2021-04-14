from rlcard.games.nano_ofcp.ofcp_utils import STRING_TO_RANK, Row
import numpy as np

class Nano_OFCP_Judger(object):
    def __init__(self, np_random):
        ''' Initialize a BlackJack judger class
        '''
        self.np_random = np_random
        
    def judge_bust(self, player):
        ''' 
        Judges whether a player is bust or not, meaning that there hands appear in not in the
        correct increasing order. Row is a wrapper around the list of cards, which provides 
        comparision methods

        Args:
            player (int): target player's id

        Returns:
            status (Bool): the status of of whether the player is Bust. 
        '''
        return Row(player.front_row) > Row(player.back_row)


    def judge_game(self, game):
        ''' 
        Judge the winner of the game. This will return the amount which should be added and 
        subtracted from a player score in the form of a tuple.
        See ReadMe.md for information about the scoring system.

        We wrap the lists of card objects into Row objects because this is where we have written the row comparitor.

        Args:
            game (class): target game class
        
        Returns:
            List[int]: [int, int] scores from this round.
        '''

        # print("Bust status. Player 1: {}, Player 2: {}".format(self.judge_bust(game.players[0]), self.judge_bust(game.players[1])))
        scores = [0, 0]
        
        if not game.is_over():
            scores = [0, 0]
        elif self.judge_bust(game.players[0]) and self.judge_bust(game.players[1]):
            scores = [0, 0]
        elif self.judge_bust(game.players[0]):
            scores = [-2, 2]
        elif self.judge_bust(game.players[1]):
            scores = [2, -2]
        else:
            # Both players in play so can compare the hands.    
            # print("Both players not bust")
            # Front row logic - if the same score then no need to move.
            player_0_front_row, player_1_front_row = Row(game.players[0].front_row), Row(game.players[1].front_row)
            # print("Front_1 type: {}".format(type(player_0_front_row)))
            # print("Front_2 type: {}".format(type(player_1_front_row)))
            # print("Front_1 < Front_2: {}".format(player_0_front_row < player_1_front_row))
            # print("Front_1 > Front_2: {}".format(player_0_front_row > player_1_front_row))
            # print("Front_1 = Front_2: {}".format(player_0_front_row == player_1_front_row))


            if player_0_front_row > player_1_front_row:
                scores = [scores[0] + 1, scores[1] - 1]
                # print("Player 1 front stronger")
            if player_0_front_row < player_1_front_row:
                scores = [scores[0] - 1, scores[1] + 1]
                # print("Player 2 front stronger")

                

            # Back row logic - if same score no need to move.
            player_0_back_row, player_1_back_row = Row(game.players[0].back_row), Row(game.players[1].back_row)
            # print("Back_1 < Back_2: {}".format(player_0_back_row < player_1_back_row))
            # print("Back_1 > Back_2: {}".format(player_0_back_row > player_1_back_row))
            # print("Back_1 = Back_2: {}".format(player_0_back_row == player_1_back_row))


            if player_0_back_row > player_1_back_row:
                scores = [scores[0] + 1, scores[1] - 1]
                # print("Player 1 back stronger")

            if player_0_back_row < player_1_back_row:
                scores = [scores[0] - 1, scores[1] + 1]
                # print("Player 2 front stronger")
        # print(scores)

        # Set the players to the scores.
        for player in game.players:
            player.score = scores[player.player_id]
        return np.array(scores)
