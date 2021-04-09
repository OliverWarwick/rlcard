import unittest
from rlcard.core import Card
from rlcard.games.nano_ofcp.ofcp_utils import Row, STRING_TO_RANK
import numpy as np


class TestOFCPUtilsComps(unittest.TestCase):

    def test_eq_three_of_kind(self):
        
        row1 = Row([Card("S", "A"), Card("H", "A"), Card("D", "A")])
        row2 = Row([Card("S", "A"), Card("H", "A"), Card("D", "A")])
        index1, hand1, rank1 = row1.evaluate_row()
        index2, hand2, rank2 = row2.evaluate_row()
        self.assertEqual(hand1, "Three of a Kind")
        self.assertEqual(hand2, "Three of a Kind")
        winner = row1 == row2
        self.assertEqual(winner, True)
    
    def test_eq_two_pair(self):
        row1 = Row([Card("S", "A"), Card("H", "J"), Card("D", "A"), Card("S", "J")])
        row2 = Row([Card("S", "A"), Card("H", "A"), Card("D", "J"), Card("C", "J")])
        index1, hand1, rank1 = row1.evaluate_row()
        index2, hand2, rank2 = row2.evaluate_row()
        self.assertEqual(hand1, "Two Pair")
        self.assertEqual(hand2, "Two Pair")
        winner = row1 == row2
        self.assertEqual(winner, True)

    def test_lt_three_of_kind(self):
        row1 = Row([Card("S", "A"), Card("H", "A"), Card("D", "A"), Card("H", "7"), Card("D", "T")])
        row2 = Row([Card("S", "A"), Card("H", "A"), Card("D", "A"), Card("H", "J"), Card("D", "8")])
        index1, hand1, rank1 = row1.evaluate_row()
        index2, hand2, rank2 = row2.evaluate_row()
        self.assertEqual(hand1, "Three of a Kind")
        self.assertEqual(hand2, "Three of a Kind")
        winner = row1 < row2
        self.assertEqual(winner, False)
    
    def test_gt_three_of_kind(self):
        row1 = Row([Card("S", "A"), Card("H", "A"), Card("D", "A"), Card("H", "J"), Card("D", "T")])
        row2 = Row([Card("S", "A"), Card("H", "A"), Card("D", "A"), Card("H", "J"), Card("D", "8")])
        index1, hand1, rank1 = row1.evaluate_row()
        index2, hand2, rank2 = row2.evaluate_row()
        self.assertEqual(hand1, "Three of a Kind")
        self.assertEqual(hand2, "Three of a Kind")
        winner = row1 < row2
        self.assertEqual(winner, False)
    
    def test_lt_pair(self):
        row1 = Row([Card("S", "5"), Card("H", "5"), Card("D", "A")])
        row2 = Row([Card("S", "5"), Card("H", "5"), Card("D", "T")])
        index1, hand1, rank1 = row1.evaluate_row()
        index2, hand2, rank2 = row2.evaluate_row()
        self.assertEqual(hand1, "One Pair")
        self.assertEqual(hand2, "One Pair")
        winner = row1 < row2
        self.assertEqual(winner, False)

    def test_four_of_kind(self):
        row1 = Row([Card("S", "5"), Card("H", "5"), Card("D", "5"), Card("C", "5")])
        row2 = Row([Card("S", "5"), Card("H", "5"), Card("D", "T"), Card("D", "T")])
        index1, hand1, rank1 = row1.evaluate_row()
        index2, hand2, rank2 = row2.evaluate_row()
        self.assertEqual(hand1, "Four of a Kind")
        self.assertEqual(hand2, "Two Pair")
        winner = row1 < row2
        self.assertEqual(winner, False)
    
    def test_two_pair_not_awk(self):
        row1 = Row([Card("S", "5"), Card("H", "5"), Card("D", "6"), Card("C", "6")])
        row2 = Row([Card("S", "5"), Card("H", "5"), Card("D", "T"), Card("D", "T")])
        index1, hand1, rank1 = row1.evaluate_row()
        index2, hand2, rank2 = row2.evaluate_row()
        self.assertEqual(hand1, "Two Pair")
        self.assertEqual(hand2, "Two Pair")
        winner = row1 < row2
        self.assertEqual(winner, True)

    def test_two_pair_awk(self):
        row1 = Row([Card("S", "5"), Card("H", "5"), Card("D", "8"), Card("C", "8"), Card("S", "7")])
        row2 = Row([Card("S", "5"), Card("H", "5"), Card("D", "T"), Card("D", "8"), Card("C", "8")])
        index1, hand1, rank1 = row1.evaluate_row()
        index2, hand2, rank2 = row2.evaluate_row()
        self.assertEqual(hand1, "Two Pair")
        self.assertEqual(hand2, "Two Pair")
        # print(row1.)
        winner = row1 < row2
        self.assertEqual(winner, True)

    def test_flush(self):
        row1 = Row([Card("S", "5"), Card("S", "6"), Card("S", "8"), Card("S", "9"), Card("S", "K")])
        row2 = Row([Card("D", "5"), Card("D", "8"), Card("D", "4"), Card("D", "8"), Card("C", "6")])
        index1, hand1, rank1 = row1.evaluate_row()
        index2, hand2, rank2 = row2.evaluate_row()
        print("Rank1: {}".format(rank1))
        self.assertEqual(hand1, "Flush")
        self.assertEqual(STRING_TO_RANK.get(rank1), 13)
        self.assertEqual(hand2, "One Pair")
        winner = row1 < row2
        self.assertEqual(winner, False)

    def test_straight(self):
        row1 = Row([Card("S", "5"), Card("D", "6"), Card("S", "8"), Card("S", "9"), Card("C", "7")])
        row2 = Row([Card("D", "7"), Card("D", "8"), Card("C", "9"), Card("D", "T"), Card("D", "J")])
        index1, hand1, rank1 = row1.evaluate_row()
        index2, hand2, rank2 = row2.evaluate_row()
        self.assertEqual(hand1, "Straight")
        self.assertEqual(STRING_TO_RANK.get(rank1), 9)
        self.assertEqual(hand2, "Straight")
        self.assertEqual(STRING_TO_RANK.get(rank2), 11)
        winner = row1 < row2
        self.assertEqual(winner, True)

    def test_straight_flush(self):
        row1 = Row([Card("S", "5"), Card("S", "6"), Card("S", "8"), Card("S", "9"), Card("S", "7")])
        row2 = Row([Card("D", "7"), Card("D", "8"), Card("D", "9"), Card("D", "T"), Card("D", "J")])
        index1, hand1, rank1 = row1.evaluate_row()
        index2, hand2, rank2 = row2.evaluate_row()
        self.assertEqual(hand1, "Straight Flush")
        self.assertEqual(STRING_TO_RANK.get(rank1), 9)
        self.assertEqual(hand2, "Straight Flush")
        self.assertEqual(STRING_TO_RANK.get(rank2), 11)
        winner = row1 < row2
        self.assertEqual(winner, True)
    
    def test_royal_flush(self):
        row1 = Row([Card("S", "T"), Card("S", "J"), Card("S", "Q"), Card("S", "K"), Card("S", "A")])
        row2 = Row([Card("D", "T"), Card("D", "J"), Card("D", "Q"), Card("D", "A"), Card("D", "K")])
        index1, hand1, rank1 = row1.evaluate_row()
        index2, hand2, rank2 = row2.evaluate_row()
        self.assertEqual(hand1, "Royal Flush")
        self.assertEqual(STRING_TO_RANK.get(rank1), 14)
        self.assertEqual(hand2, "Royal Flush")
        self.assertEqual(STRING_TO_RANK.get(rank2), 14)
        winner = row1 == row2
        self.assertEqual(winner, True)


    
    def test_front_back(self):
        front_1 = Row([Card(rank='9', suit='D'), Card(rank='9', suit='C'), Card(rank='9', suit='S')])
        front_2 = Row([Card(rank='J', suit='D'), Card(rank='J', suit='C'), Card(rank='J', suit='S')])
        back_1 = Row([Card(rank='T', suit='D'), Card(rank='T', suit='C'), Card(rank='T', suit='S')])
        back_2 = Row([Card(rank='Q', suit='D'), Card(rank='Q', suit='C'), Card(rank='Q', suit='S')])
        winner_1 = front_1 < front_2
        winner_2 = back_1 < back_2
        self.assertTrue(winner_1)
        self.assertTrue(winner_2)
    

    


if __name__ == '__main__':
    unittest.main()