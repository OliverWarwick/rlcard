import numpy as np
import random 
import math 
import time
from rlcard.games.nano_ofcp.ofcp_utils import STRING_TO_RANK, Row
from rlcard.utils.utils import init_mini_deck

''' TODO: '''


class SimpleAgent:

    def __init__(self, action_num, use_raw):
        self.use_raw = use_raw
        self.action_num = action_num

        # todo
        self.possible_ending_hands = [[['High Card', 'T'], ['High Card', 'J']], [['High Card', 'T'], ['High Card', 'Q']], [['High Card', 'T'], ['High Card', 'K']], [['High Card', 'T'], ['High Card', 'A']], [['High Card', 'T'], ['One Pair', 'T']], [['High Card', 'T'], ['One Pair', 'J']], [['High Card', 'T'], ['One Pair', 'Q']], [['High Card', 'T'], ['One Pair', 'K']], [['High Card', 'T'], ['One Pair', 'A']], [['High Card', 'T'], ['Three of a Kind', 'T']], [['High Card', 'T'], ['Three of a Kind', 'J']], [['High Card', 'T'], ['Three of a Kind', 'Q']], [['High Card', 'T'], ['Three of a Kind', 'K']], [['High Card', 'T'], ['Three of a Kind', 'A']], [['High Card', 'J'], ['High Card', 'Q']], [['High Card', 'J'], ['High Card', 'K']], [['High Card', 'J'], ['High Card', 'A']], [['High Card', 'J'], ['One Pair', 'T']], [['High Card', 'J'], ['One Pair', 'J']], [['High Card', 'J'], ['One Pair', 'Q']], [['High Card', 'J'], ['One Pair', 'K']], [['High Card', 'J'], ['One Pair', 'A']], [['High Card', 'J'], ['Three of a Kind', 'T']], [['High Card', 'J'], ['Three of a Kind', 'J']], [['High Card', 'J'], ['Three of a Kind', 'Q']], [['High Card', 'J'], ['Three of a Kind', 'K']], [['High Card', 'J'], ['Three of a Kind', 'A']], [['High Card', 'Q'], ['High Card', 'K']], [['High Card', 'Q'], ['High Card', 'A']], [['High Card', 'Q'], ['One Pair', 'T']], [['High Card', 'Q'], ['One Pair', 'J']], [['High Card', 'Q'], ['One Pair', 'Q']], [['High Card', 'Q'], ['One Pair', 'K']], [['High Card', 'Q'], ['One Pair', 'A']], [['High Card', 'Q'], ['Three of a Kind', 'T']], [['High Card', 'Q'], ['Three of a Kind', 'J']], [['High Card', 'Q'], ['Three of a Kind', 'Q']], [['High Card', 'Q'], ['Three of a Kind', 'K']], [['High Card', 'Q'], ['Three of a Kind', 'A']], [['High Card', 'K'], ['High Card', 'A']], [['High Card', 'K'], ['One Pair', 'T']], [['High Card', 'K'], ['One Pair', 'J']], [['High Card', 'K'], ['One Pair', 'Q']], [['High Card', 'K'], ['One Pair', 'K']], [['High Card', 'K'], ['One Pair', 'A']], [['High Card', 'K'], ['Three of a Kind', 'T']], [['High Card', 'K'], ['Three of a Kind', 'J']], [['High Card', 'K'], ['Three of a Kind', 'Q']], [['High Card', 'K'], ['Three of a Kind', 'K']], [['High Card', 'K'], ['Three of a Kind', 'A']], [['High Card', 'A'], ['One Pair', 'T']], [['High Card', 'A'], ['One Pair', 'J']], [['High Card', 'A'], ['One Pair', 'Q']], [['High Card', 'A'], ['One Pair', 'K']], [['High Card', 'A'], ['One Pair', 'A']], [['High Card', 'A'], ['Three of a Kind', 'T']], [['High Card', 'A'], ['Three of a Kind', 'J']], [['High Card', 'A'], ['Three of a Kind', 'Q']], [['High Card', 'A'], ['Three of a Kind', 'K']], [['High Card', 'A'], ['Three of a Kind', 'A']], [['One Pair', 'T'], ['One Pair', 'J']], [['One Pair', 'T'], ['One Pair', 'Q']], [['One Pair', 'T'], ['One Pair', 'K']], [['One Pair', 'T'], ['One Pair', 'A']], [['One Pair', 'T'], ['Three of a Kind', 'J']], [['One Pair', 'T'], ['Three of a Kind', 'Q']], [['One Pair', 'T'], ['Three of a Kind', 'K']], [['One Pair', 'T'], ['Three of a Kind', 'A']], [['One Pair', 'J'], ['One Pair', 'Q']], [['One Pair', 'J'], ['One Pair', 'K']], [['One Pair', 'J'], ['One Pair', 'A']], [['One Pair', 'J'], ['Three of a Kind', 'T']], [['One Pair', 'J'], ['Three of a Kind', 'Q']], [['One Pair', 'J'], ['Three of a Kind', 'K']], [['One Pair', 'J'], ['Three of a Kind', 'A']], [['One Pair', 'Q'], ['One Pair', 'K']], [['One Pair', 'Q'], ['One Pair', 'A']], [['One Pair', 'Q'], ['Three of a Kind', 'T']], [['One Pair', 'Q'], ['Three of a Kind', 'J']], [['One Pair', 'Q'], ['Three of a Kind', 'K']], [['One Pair', 'Q'], ['Three of a Kind', 'A']], [['One Pair', 'K'], ['One Pair', 'A']], [['One Pair', 'K'], ['Three of a Kind', 'T']], [['One Pair', 'K'], ['Three of a Kind', 'J']], [['One Pair', 'K'], ['Three of a Kind', 'Q']], [['One Pair', 'K'], ['Three of a Kind', 'A']], [['One Pair', 'A'], ['Three of a Kind', 'T']], [['One Pair', 'A'], ['Three of a Kind', 'J']], [['One Pair', 'A'], ['Three of a Kind', 'Q']], [['One Pair', 'A'], ['Three of a Kind', 'K']], [['Three of a Kind', 'T'], ['Three of a Kind', 'J']], [['Three of a Kind', 'T'], ['Three of a Kind', 'Q']], [['Three of a Kind', 'T'], ['Three of a Kind', 'K']], [['Three of a Kind', 'T'], ['Three of a Kind', 'A']], [['Three of a Kind', 'J'], ['Three of a Kind', 'Q']], [['Three of a Kind', 'J'], ['Three of a Kind', 'K']], [['Three of a Kind', 'J'], ['Three of a Kind', 'A']], [['Three of a Kind', 'Q'], ['Three of a Kind', 'K']], [['Three of a Kind', 'Q'], ['Three of a Kind', 'A']], [['Three of a Kind', 'K'], ['Three of a Kind', 'A']]]
        
        self.hand_values = {'High Card': 0, 'One Pair': 1, 'Three of a Kind':2}
        self.card_values = {'T':0, 'J':0.2, 'Q':0.4, 'K':0.6, 'A':0.8}
    

    def step(self, state, env, player_id): 
        # Going to use the environment to step forwards and back based on the current actions.

        # Initialise to defaults.
        best_reward, best_action = -2, state['legal_actions'][0]

        actions = state['legal_actions']
        for action in actions:
            # Step fowards using the action
            env.step(action)
            cards_already_dealt = dict()

            # Compute the dictionary of how many of each card are around
            for row in [env.players[player_id].front_row, env.players[player_id].back_row, env.players[player_id].discard_pile, env.players[(player_id + 1) % len(env.players)].front_row, env.players[(player_id + 1) % len(env.players)].back_row]:
                for card in row:
                    if card.rank in cards_already_dealt:
                        cards_already_dealt[card.rank] = cards_already_dealt[card.rank] + 1
                    else:
                        cards_already_dealt[card.rank] = 1

            # Cycle through the ending hands estimating the prob of each
            for possible_ending in self.possible_ending_hands:
                
                print("Ending: {}".format(possible_ending))
                prob_complete_hand = 0.0
                print("Prob: {}".format(prob_complete_hand))
                value_complete_hand = self.eval_value_hand(possible_ending)
                print("Value: {}".format(value_complete_hand))
                e_v = prob_complete_hand * value_complete_hand

                if e_v > best_reward:
                    best_reward, best_action = e_v, action
                    # Add to debug the possible ending
                    print("Updating best action. New ending with highest EV: {}, Action: {}, EV: {}".format(possible_ending, action, e_v))
            
            # Step back to original state
            env.step_back()
            # Reset the dictionary used

        print("Return action: {}".format(best_action))
        return best_action

    def eval_step(self, state, env, player_id):
        return self.step(state, env, player_id)


    def eval_value_hand(self, possible_ending):
        front_hand_type, back_hand_type = possible_ending[0][0], possible_ending[1][0]
        front_card_type, back_card_type = possible_ending[0][1], possible_ending[1][1]

        return self.hand_values.get(front_hand_type) + self.hand_values.get(back_hand_type) + self.card_values.get(front_card_type) + self.card_values.get(back_card_type)

    def eval_prob_hand(self, env, player_id, cards_already_dealt, possible_ending):
        number_cards_left_to_deal = (4 - env.game.dealer.deal_counter) * 2  # As we have to throw away one of these cards each time. This is rough probs, not the actual one.
        number_of_cards_dealt = env.game.dealer.deal_counter * 2 * 3 # Two for each player, 3 for each deal.
        print("Cards dealt: {}   Number of deals left: {}".format(number_of_cards_dealt, number_cards_left_to_deal))
        front_hand_type, back_hand_type = possible_ending[0][0], possible_ending[1][0]
        front_card_type, back_card_type = possible_ending[0][1], possible_ending[1][1]

        cards_map = {"High Card": 1, "One Pair": 2, "Three of a Kind": 3}

        cards_required_front = cards_map.get(front_hand_type) - cards_already_dealt.get(front_card_type)
        cards_required_back = cards_map.get(back_hand_type) - cards_already_dealt.get(back_card_type)
        if cards_required_front < 0 or cards_required_back < 0 or cards_required_front + cards_required_back > number_cards_left_to_deal:
            return 0.0
        
        # TODO: NEED TO DO A BIT MORE READING.

        # Total number of possible allocation.
        total_combinations = math.comb(20 - number_of_cards_dealt, number_cards_left_to_deal) * math.factorial(number_cards_left_to_deal)

        # Number of allocation which satify what we need.
        # Pick front first, then back.
        our_combinations = math.comb(4 - cards_already_dealt.get(front_card_type), cards_required_front) * math.comb(number_cards_left_to_deal, cards_required_front) * \
            math.comb(4 - cards_already_dealt.get(back_card_type), cards_required_back) * math.comb(number_cards_left_to_deal - cards_required_front, cards_required_back)
        

        








if __name__ == '__main__':

    # endings = {'High Card': 0, 'One Pair': 1, 'Three of a Kind': 2}
    hands = {'High Card': 0, 'One Pair': 1, 'Three of a Kind':2}
    cards = {'T':0, 'J':0.2, 'Q':0.4, 'K':0.6, 'A':0.8}

    endings = []

    for front, front_value in hands.items():
        for front_high_card, front_card_value in cards.items():
            for back, back_value in hands.items():
                for back_high_card, back_card_value in cards.items():
                    if front_value < back_value or front_value == back_value and front_card_value < back_card_value:
                        if front == 'One Pair' and back == 'Three of a Kind' and front_high_card == back_high_card or front == 'Three of a Kind' and back == 'One Pair' and front_high_card == back_high_card:
                            continue
                        else:
                            endings.append([[front, front_high_card], [back, back_high_card]])
    
    print(endings)
    # print(len(endings))




