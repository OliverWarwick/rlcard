import json
import os
import numpy as np
from copy import deepcopy

import rlcard
from rlcard.envs import Env
from rlcard.games.nano_ofcp import Game, ActionChoiceException
from rlcard.games.nano_ofcp.ofcp_utils import STRING_TO_RANK
from rlcard.agents import RandomAgent, NanoOFCPPerfectInfoAgent, MonteCarloAgent, DQNAgentEndGame
from rlcard.utils import *

DEFAULT_GAME_CONFIG = {
    'game_player_num': 2,
}

class NanoOFCPEnv(Env):

    ''' 
    Nano OFCP Environment.
    This is the outward facing interface, and will call the  underlying methods form within game. 
    This yields the functions get_payoffs, get_perfect_infomation.
    Run using the following signature: 

    env = rlcard.make('blackjack')
    env.set_agents([RandomAgent(action_num=env.action_num)])
    trajectories, payoffs = env.run()

    This makes calls from the Env super class to our version of game.


    State Encoding:
    As this is a 3 card version of OFCP suits are not important. By this I mean, if both players have triples there is no benefit to one suit over another, whereas in normal OFCP one player may have a straight flush and the other just a straight.
    As a result for each row we moniter:
        Our_Card_To_Play, Our_Front, Our_Back, Our_Discard, Oppo_Front, Oppo_Back
    we add a length 3 vector which takes the cards inside [which could be of the special int = 0 if the slot is empty] and takes the numeric value of the rank of this card e.g: J == 11, T == 10 etc...]
    e.g [[None, None, None,], [TD, TC, None], .. ] -> [0, 0, 0, 10, 10, 0, ...]
    '''

    def __init__(self, config):
        ''' Initalise the environment '''
        self.name = 'nano-ofcp'
        self.default_game_config = DEFAULT_GAME_CONFIG
        self.game = Game()
        super().__init__(config)
        self.actions = [['D', 'F', 'F'], ['D', 'F', 'B'], ['D', 'B', 'F'], ['D', 'B', 'B'], ['F', 'D', 'F'], ['F', 'D', 'B'], ['B', 'D', 'F'], ['B', 'D', 'B'], ['F', 'F', 'D'], ['F', 'B', 'D'], ['B', 'F', 'D'], ['B', 'B', 'D']]
        self.state_shape = 6 * 3 * 6 # Each row is 3 cards and their are 6 of these. For each of the cards we have a 6 element one hot encoding.



    def end_game_agent_run(self, is_training=False):

        # Need a special routine for the mc agent because it requires the env and player_id alongside the state when choosing an action.
        trajectories = [[] for _ in range(self.player_num)]
        state, player_id = self.reset()

        trajectories[player_id].append(state)
        while not self.is_over():
            # print("Self game pointer: " + str(self.game.game_pointer))

            # Agent plays
            if isinstance(self.agents[player_id], DQNAgentEndGame):
                action = self.agents[player_id].step(state, self, player_id)
            else:
                if not is_training:
                    # print("Calling Random Agent Step")
                    action, _ = self.agents[player_id].eval_step(state)
                else:
                    action = self.agents[player_id].step(state)

            # Environment steps
            next_state, next_player_id = self.step(action, self.agents[player_id].use_raw)
            # Save action
            trajectories[player_id].append(action) 
            # Set the state and player
            state = next_state
            player_id = next_player_id

            # Save state.
            if not self.game.is_over():
                trajectories[player_id].append(state)

        # Add a final state to all the players
        for player_id in range(self.player_num):
            state = self.get_state(player_id)
            trajectories[player_id].append(state)

        # Payoffs
        payoffs = self.get_payoffs()

        # Reorganize the trajectories
        trajectories = reorganize(trajectories, payoffs)

        return trajectories, payoffs




    def mc_agent_run(self, is_training=False):

        # Need a special routine for the mc agent because it requires the env and player_id alongside the state when choosing an action.
        trajectories = [[] for _ in range(self.player_num)]
        state, player_id = self.reset()

        trajectories[player_id].append(state)
        while not self.is_over():
            # print("Self game pointer: " + str(self.game.game_pointer))

            # Agent plays
            if isinstance(self.agents[player_id], MonteCarloAgent):
                action = self.agents[player_id].step(state, self, player_id)
            else:
                if not is_training:
                    # print("Calling Random Agent Step")
                    action, _ = self.agents[player_id].eval_step(state)
                else:
                    action = self.agents[player_id].step(state)

            # Environment steps
            next_state, next_player_id = self.step(action, self.agents[player_id].use_raw)
            # Save action
            trajectories[player_id].append(action) 
            # Set the state and player
            state = next_state
            player_id = next_player_id

            # Save state.
            if not self.game.is_over():
                trajectories[player_id].append(state)

        # Add a final state to all the players
        for player_id in range(self.player_num):
            state = self.get_state(player_id)
            trajectories[player_id].append(state)

        # Payoffs
        payoffs = self.get_payoffs()

        # Reorganize the trajectories
        trajectories = reorganize(trajectories, payoffs)

        return trajectories, payoffs


    def heuristic_agent_run(self, is_training=False):
        '''
        Run a complete game, either for evaluation or training RL agent.

        Args:
            is_training (boolean): True if for training purpose.

        Returns:
            (tuple) Tuple containing:
            (list): A list of trajectories generated from the environment.
            (list): A list payoffs. Each entry corresponds to one player.

        Note: The trajectories are 3-dimension list. The first dimension is for different players.
              The second dimension is for different transitions. The third dimension is for the contents of each transiton
        '''
        trajectories = [[] for _ in range(self.player_num)]
        state, player_id = self.reset()

        # Once we've reset ontain the cards which will be dealt. 
        # print("Cards in the deck after reset (init_game called): ")
        # print(self.game.dealer.deck)
        # print("Cards in the players hands: ")
        # print(self.game.players[0].cards_to_play)
        # print(self.game.players[1].cards_to_play)
        # print("Game pointer: ", self.game.game_pointer)

        # Check whether the Heurisitc agent is the first or the second character. 
        if isinstance(self.agents[0], NanoOFCPPerfectInfoAgent):
            dealt_cards = [0,1,2,6,7,8]
            cards_to_deal = [cards for index, cards in enumerate(self.game.dealer.deck) if index in dealt_cards]
            # print("Cards to deal to heuristic agent: " + str(cards_to_deal))
            
            # print("Len to play: ", len(self.game.players[0].cards_to_play))
            # print("Len of cards: " + str(len(cards_to_deal)))
            self.agents[0].set_up(deals=self.game.players[0].cards_to_play + cards_to_deal)
        if isinstance(self.agents[1], NanoOFCPPerfectInfoAgent):
            dealt_cards = [3,4,5,9,10,11]
            cards_to_deal = [cards for index, cards in enumerate(self.game.dealer.deck) if index in dealt_cards]
            # print("Cards to deal to heuristic agent: " + str(cards_to_deal))
            self.agents[1].set_up(deals=self.game.players[1].cards_to_play + cards_to_deal)
    

        # Loop to play the game
        trajectories[player_id].append(state)
        while not self.is_over():
            # print("Self game pointer: " + str(self.game.game_pointer))

            # Agent plays
            if isinstance(self.agents[player_id], NanoOFCPPerfectInfoAgent):
                if not is_training:
                    # print("Calling heuristic agent eval_step")
                    # print("Cards to play for heuristic agent: " + str(self.game.players[player_id].cards_to_play))
                    action = self.agents[player_id].eval_step()
                else:
                    action = self.agents[player_id].step()
            else:
                if not is_training:
                    # print("Calling random agent eval_step")
                    action, _ = self.agents[player_id].eval_step(state)
                else:
                    action = self.agents[player_id].step(state)

            # Environment steps
            next_state, next_player_id = self.step(action, self.agents[player_id].use_raw)
            # Save action
            trajectories[player_id].append(action) 
            # Set the state and player
            state = next_state
            player_id = next_player_id

            # Save state.
            if not self.game.is_over():
                trajectories[player_id].append(state)

        # Add a final state to all the players
        for player_id in range(self.player_num):
            state = self.get_state(player_id)
            trajectories[player_id].append(state)

        # Payoffs
        payoffs = self.get_payoffs()

        # Reorganize the trajectories
        trajectories = reorganize(trajectories, payoffs)

        return trajectories, payoffs



    def _get_legal_actions(self):
        ''' Get all leagal actions

        Returns:
            encoded_action_list (list): return encoded legal action list (from str to int)
        '''
        return self.game.get_legal_actions()

    def _extract_state(self, state):
        ''' Extract the state representation from state dictionary for agent

        Note: Currently the use the hand cards and the public cards. TODO: encode the states

        Args:
            state (dict): Original state from the game

        Returns:
            observation (list): combine the player's score and dealer's observable score for observation
        '''
        extracted_state = {}
        legal_actions = [self.actions.index(a) for a in state['legal_actions']]
        extracted_state['legal_actions'] = legal_actions   

        # Processing the card information.
        extracted_state['hands'] = deepcopy(state['state'][0] + state['state'][1])

        for row in extracted_state['hands']:
            for _ in range(3-len(row)):
                row.append(None)

        # print(extracted_state['hands'])
        # Let f be a function which maps a vector of cards e.g: [TD, QS, None] to a one encoding of each card.
        # There are 6 possiblilities [None, T, J, Q, K, A] which will used to select the one hot element.
        one_hot_mapping = {"T": 1, "J": 2, "Q": 3, "K": 4, "A": 5}
        # f = lambda x: STRING_TO_RANK.get(x.rank) if x is not None else 0
        index_vector = np.array([[one_hot_mapping.get(x.rank) if x is not None else 0 for x in row] for row in extracted_state['hands']]).flatten()
        # print(index_vector)
        # Then reshape this into a one - hot vector.
        assert len(index_vector) == 18
        one_hot_matrix = np.zeros((index_vector.size, 6))
        one_hot_matrix[np.arange(index_vector.size), index_vector] = 1
        # print(one_hot_matrix)
        assert np.shape(one_hot_matrix) == (18, 6)
        one_hot_vector = one_hot_matrix.flatten()
        assert np.shape(one_hot_vector) == (18 * 6, )
        # print(one_hot_vector)


        # We want to run through these and anywhere there is empty space we can add
        # extra zeros from the index. 
        # Obvs is what an agent can use to make a decision. This in our case is the vector embedding of our cards and those which are visable from the side of the oppo.
        extracted_state['obs'] = one_hot_vector

        # We pass back the raw data which is used by functions such as the human player to write out all of the cards. 
        if self.allow_raw_data:
            extracted_state['raw_obs'] = state
            extracted_state['raw_legal_actions'] = [a for a in state['legal_actions']]
        # TODO: Figure out what this does.
        if self.record_action:
            extracted_state['action_record'] = self.action_recorder

        return extracted_state

    def get_payoffs(self):
        ''' Get the payoff of a game

        Returns:
           payoffs (list): list of payoffs
        '''
        return self.game.get_payoffs()

    def _decode_action(self, action_id):

        legal_actions = self.game.get_legal_actions()
        # print("Legal actions: {}".format(legal_actions))
        if self.actions[action_id] in legal_actions:
            return self.actions[action_id]
        else:
            # Raise an exception for now we can figure this out later.
            raise ActionChoiceException("Tried to call action {} which is not possible given game state.".format(self.actions[action_id]))


    def _load_model(self):
        pass

    def get_perfect_information(self):
        ''' Get the perfect information of the current state. We can do this by using the game field and etraing all information required.
        
        Returns:
            (dict): A dictionary of all the perfect information of the current state
        '''

        state = {}
        state['player_0_hand'] = [self.game.players[0].cards_to_play, self.game.players[0].front_row, self.game.players[0].back_row, self.game.players[0].discard_pile]
        state['player_1_hand'] = [self.game.players[1].cards_to_play, self.game.players[1].front_row, self.game.players[1].back_row, self.game.players[1].discard_pile]
        state['scores'] = [self.game.players[0].score, self.game.players[1].score]
        state['current_player'] = self.game.game_pointer
        state['legal_actions'] = self.game.get_legal_actions()
        return state