import unittest
import numpy as np

from rlcard.games.nano_ofcp.game import Nano_OFCP_Game as Game
from rlcard.envs.nano_ofcp import DEFAULT_GAME_CONFIG
from rlcard.core import Card

class TestNanoOFCPGame(unittest.TestCase):

    def test_get_action_num(self):
        game = Game()
        game.configure(DEFAULT_GAME_CONFIG)
        action_num = game.get_action_num()
        self.assertEqual(action_num, 12)
    
    def test_init_game(self):
        game = Game()
        game.configure(DEFAULT_GAME_CONFIG)
        state, current_player = game.init_game()
        self.assertEqual(len(game.history), 0)
        self.assertEqual(current_player, 0)
        self.assertEqual(game.players[0].score, 0)
        # self.assertEqual(len(state['state'][0]), len(state['state'][1])+1)

    def test_step(self):
        game = Game()
        game.configure(DEFAULT_GAME_CONFIG)
        game.init_game()
        next_state, next_player = game.step(['D', 'F', 'B'])
        self.assertTrue(game.players[0].front_row != [] or game.players[0].back_row != [])
        self.assertTrue(game.players[1].front_row == [] and game.players[1].back_row == [])
        # Test TODO
        next_state, _ = game.step(['D', 'F', 'B'])
        self.assertTrue(game.players[1].front_row != [] or game.players[1].back_row != [])        # Test TODO

    def test_get_state(self):
        game = Game()
        game.configure(DEFAULT_GAME_CONFIG)
        game.init_game()
        # Check that the first set of cards to play is not empty for both players.
        self.assertTrue(game.get_state(0)['my_hand'][0][0] is not None)
        self.assertTrue(game.get_state(1)['my_hand'][0][0] is not None)
        next_state, _ = game.step(['D', 'F', 'B'])
        self.assertTrue(game.get_state(0)['my_hand'][1][0] is not None)
        self.assertTrue(game.get_state(1)['my_hand'][0][0] is not None)
        next_state, _ = game.step(['D', 'F', 'B'])
        self.assertTrue(game.get_state(0)['my_hand'][1][0] is not None)
        self.assertTrue(game.get_state(1)['my_hand'][1][0] is not None)

    
    def test_reward_mid_game(self):
        game = Game()
        game.configure(DEFAULT_GAME_CONFIG)
        game.init_game()
        payoff = game.get_payoffs()
        self.assertEqual(payoff, [0, 0]) # May change if we use numpy arrays.

    def test_reward_end_game(self):
        game = Game()
        game.configure(DEFAULT_GAME_CONFIG)
        game.init_game()
        # Set conditions so game over is satified.
        game.dealer.deal_counter = 3
        game.round.players_finished = [True, True]
        
        # Assert game is done
        self.assertTrue(game.is_over())
        
        # Set the hands.    
        game.players[0].front_row = [Card(rank='9', suit='D'), Card(rank='9', suit='C'), Card(rank='9', suit='S')]
        game.players[1].front_row = [Card(rank='J', suit='D'), Card(rank='J', suit='C'), Card(rank='J', suit='S')]
        game.players[0].back_row = [Card(rank='T', suit='D'), Card(rank='T', suit='C'), Card(rank='T', suit='S')]
        game.players[1].back_row = [Card(rank='Q', suit='D'), Card(rank='Q', suit='C'), Card(rank='Q', suit='S')]
        payoff = game.get_payoffs()
        self.assertListEqual(payoff, [-2, 2])

        game.players[0].front_row = [Card(rank='9', suit='D'), Card(rank='T', suit='C'), Card(rank='J', suit='S')]
        game.players[1].front_row = [Card(rank='J', suit='D'), Card(rank='J', suit='C'), Card(rank='J', suit='S')]
        game.players[0].back_row = [Card(rank='9', suit='D'), Card(rank='K', suit='C'), Card(rank='K', suit='S')]
        game.players[1].back_row = [Card(rank='Q', suit='D'), Card(rank='Q', suit='C'), Card(rank='Q', suit='S')]
        payoff = game.get_payoffs()
        self.assertListEqual(payoff, [-2, 2])

        game.players[0].front_row = [Card(rank='T', suit='D'), Card(rank='T', suit='C'), Card(rank='9', suit='S')]
        game.players[1].front_row = [Card(rank='T', suit='S'), Card(rank='T', suit='H'), Card(rank='9', suit='S')]
        game.players[0].back_row = [Card(rank='Q', suit='D'), Card(rank='Q', suit='C'), Card(rank='K', suit='S')]
        game.players[1].back_row = [Card(rank='Q', suit='S'), Card(rank='Q', suit='H'), Card(rank='K', suit='D')]
        payoff = game.get_payoffs()
        self.assertListEqual(payoff, [0, 0])

        game.players[0].front_row = [Card(rank='T', suit='D'), Card(rank='T', suit='C'), Card(rank='5', suit='S')]
        game.players[1].front_row = [Card(rank='T', suit='S'), Card(rank='T', suit='H'), Card(rank='9', suit='S')]
        game.players[0].back_row = [Card(rank='Q', suit='D'), Card(rank='Q', suit='C'), Card(rank='A', suit='S')]
        game.players[1].back_row = [Card(rank='Q', suit='S'), Card(rank='Q', suit='H'), Card(rank='K', suit='D')]
        payoff = game.get_payoffs()
        self.assertListEqual(payoff, [0, 0])


if __name__ == '__main__':
    unittest.main()



    
