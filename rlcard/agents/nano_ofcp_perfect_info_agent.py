import numpy as np
import random 
from rlcard.games.nano_ofcp.ofcp_utils import STRING_TO_RANK, Row
from rlcard.utils.utils import init_mini_deck
import math 
import time
''' This will use a perfect info set of cards to decide how to move. 
The heurstic used is the rank of the poker hands for front and back and their rank of the highest card. This (may or may not be) optimal - i think not - because in order to be the best solution to placement it must beat the previous best in all categores, i.e: 2 pair is not replaced with triple and single card. # TODO: Play with this '''


class PerfectInfoAgent:


    def __init__(self, action_num, use_raw, alpha=1.0):

        self.use_raw = use_raw
        self.action_num = action_num
        self.actions = [['D', 'F', 'F'], ['D', 'F', 'B'], ['D', 'B', 'F'], ['D', 'B', 'B'], ['F', 'D', 'F'], ['F', 'D', 'B'], ['B', 'D', 'F'], ['B', 'D', 'B'], ['F', 'F', 'D'], ['F', 'B', 'D'], ['B', 'F', 'D'], ['B', 'B', 'D']]
        self.mock_front_row = []
        self.mock_back_row = []
        self.queued_actions = []
        self.alpha = alpha
        self.explored_percentage = 0.0
        self.set_up_called = 0

    def set_up(self, deals):

        # print("Calling setup")
        # Need to sort so these appear in the same order which they will be dealt in as the dealer imposes an order over these.
        # This will ensure that the actions line up properly 
        deal1, deal2, deal3 = sorted(deals[0:3], key=lambda card: (STRING_TO_RANK.get(card.rank), card.suit), reverse=True), sorted(deals[3:6], key=lambda card: (STRING_TO_RANK.get(card.rank), card.suit), reverse=True), sorted(deals[6:9], key=lambda card: (STRING_TO_RANK.get(card.rank), card.suit), reverse=True)
        # print(deal1, deal2, deal3)
        # start = time.time()
        self.set_up_called += 1
        self.eval_strategies(deal1, deal2, deal3)
        # print("Time taken: {}".format(time.time() - start))


    def step(self):

        # deals should be a full list of the deals, so we split it here.
        # print("Queued actions: ", self.queued_actions)
        action = self.queued_actions.pop(0)
        return action

    def eval_step(self):
        # As there is no training we can just called the step method.
        return self.step()

    
    def eval_strategies(self, deal1, deal2, deal3):

        # print(deal1, deal2, deal3)
        # Set the final action to be an arbitrary action which we know will allow play, but is not
        # always going to give a 'non-bust' outcome. This is for use in the case where we are 
        # sampling at the low alphas (e.g: 0.1) in which we may get no non-bust solutions in our
        # sample set.

        final_action = [['D', 'F', 'B'], ['D', 'F', 'B'], ['D', 'F', 'B']]
        actions_checked = 0

        front_lowest_poker_index = 9 
        front_highest_card_rank = 0

        back_lowest_poker_index = 9
        back_highest_card_rank = 0

        # Sample actions from the set to explore, these will all be legal so we can explore all.

        # sampled_first_actions = np.array(self.actions)[np.random.choice(len(self.actions), int(alpha * len(self.actions)), replace=False)]
        sampled_first_actions = random.sample(self.actions, max(1, math.ceil(self.alpha * len(self.actions))))
        # print(sampled_first_actions)
        # print("Len of original list: {}, new calc number: {}, new list length: {}".format(len(self.actions), str(max(1, math.ceil(alpha * len(self.actions)))), len(sampled_first_actions)))
        #print("First action: " + str(sampled_first_actions))
        for action1 in sampled_first_actions:
            self.add_to_hand(action1, deal1)

            full_second_actions = self.get_legal_actions()
            sampled_second_actions = random.sample(full_second_actions, max(1, math.ceil(self.alpha * len(full_second_actions))))
            # print(sampled_second_actions)
            # print("Len of original list: {}, new calc number: {}, new list length: {}".format(len(full_second_actions), str(max(1, int(alpha * len(full_second_actions)))), len(sampled_second_actions)))
            # print("Second action: " + str(sampled_second_actions))
            for action2 in sampled_second_actions:
                self.add_to_hand(action2, deal2)

                full_third_actions = self.get_legal_actions()
                sampled_third_actions = random.sample(full_third_actions, max(1, math.ceil(self.alpha * len(full_third_actions))))
                # print(sampled_third_actions)
                # print("Third action: " + str(sampled_third_actions))
                # print("Len of original list: {}, new calc number: {}, new list length: {}".format(len(full_third_actions), str(max(1, int(alpha * len(full_third_actions)))), len(sampled_third_actions)))
                
                for action3 in sampled_third_actions:
                    self.add_to_hand(action3, deal3)
                    actions_checked += 1

                    front_row = Row(self.mock_front_row)
                    back_row = Row(self.mock_back_row)

                    # Check that this does not yield a bust hand.
                    if front_row < back_row:
                        front_poker_index, _, front_rank = front_row.evaluate_row()
                        back_poker_index, _, back_rank = back_row.evaluate_row()

                        front_card_rank = STRING_TO_RANK.get(front_rank)
                        back_card_rank = STRING_TO_RANK.get(back_rank)

                        # Here we want one or 4 things to be true. Either: 
                        #  - Front index is better, and back is no worse for index and rank.
                        #  - Back index is better, and front is no worse for index and rank.
                        #  - Front index is the same, but rank inproved. Back is no worse for index and rank.
                        #  - Back index is the same, but rank improved. Front is no worse for index and rank.
                        
                        if (front_poker_index < front_lowest_poker_index and back_poker_index <= back_lowest_poker_index and back_card_rank >= back_highest_card_rank) or \
                        (back_poker_index < back_lowest_poker_index and front_poker_index <= front_lowest_poker_index and front_card_rank >= front_highest_card_rank) or \
                        (front_poker_index == front_lowest_poker_index and front_card_rank > front_highest_card_rank and back_poker_index <= back_lowest_poker_index and back_card_rank >= back_highest_card_rank) or \
                        (back_poker_index == back_lowest_poker_index and back_card_rank > back_highest_card_rank and front_poker_index <= front_lowest_poker_index and front_card_rank >= front_highest_card_rank):

                            # print("Replacing best action with: ")
                            final_action = [action1, action2, action3]
                            # print(final_action)
                            front_lowest_poker_index, front_highest_card_rank, back_lowest_poker_index, back_highest_card_rank = front_poker_index, front_card_rank, back_poker_index, back_card_rank
                            # print("Front Row: {}. \nBack Row: {}".format(self.mock_front_row, self.mock_back_row))

                    self.remove_from_hand(action3, deal3)
                
                self.remove_from_hand(action2, deal2)
            
            self.remove_from_hand(action1, deal1)
        
        print("Actions chekcked: " + str(actions_checked) + "  Percentage of full action space: " + str(actions_checked / 540))
        # print("Best set of actions: ")
        # print(final_action)
        # print("Front_index: {}, Front_rank: {}, Back_index: {}, Back_rank: {}".format(front_lowest_index, front_highest_rank, back_lowest_index, back_highest_rank))
        # print("deal1: {} deal2: {} deal3: {}".format(deal1, deal2, deal3))
        self.queued_actions = final_action
        self.explored_percentage = self.explored_percentage + ((actions_checked / 540) - self.explored_percentage) / self.set_up_called




    def add_to_hand(self, action, deal):
        for act, card in zip(action, deal):
            if act == 'F':
                self.mock_front_row.append(card)
            if act == 'B':
                self.mock_back_row.append(card)

    def remove_from_hand(self, action, deal):
        for act, card in zip(action, deal):
            if act == 'F':
                self.mock_front_row.remove(card)
            if act == 'B':
                self.mock_back_row.remove(card)

    def get_legal_actions(self):
        
        # Use the game pointer to get the player and then remove which of the tests get failed.

        return [action for action in self.actions if (action.count('F') + len(self.mock_front_row) <= 3) and (action.count('B') + len(self.mock_back_row) <= 3)]

        
# if __name__ == '__main__':
#     agent = PerfectInfoAgent(action_num=12)
#     print(agent.get_legal_actions())

#     deck = init_mini_deck()

#     row1 = [deck[0], deck[5], deck[9]]
#     row2 = [deck[11], deck[12], deck[13]]
#     row3 = [deck[17], deck[14], deck[19]]

#     agent.eval_strategies(row1, row2, row3)
