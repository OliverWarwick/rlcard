''' DQN agent

The code is derived from https://github.com/dennybritz/reinforcement-learning/blob/master/DQN/dqn.py

Copyright (c) 2019 Matthew Judell
Copyright (c) 2019 DATA Lab at Texas A&M University
Copyright (c) 2016 Denny Britz

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''



'''
OW - This should work as  DQN Neg Reward agent, apart from the last move, which we solve, place into a hashmap
and then we can retreive this whenever required in the future when that move is encountered. 

For now we model the opponent as random, but we could search for the expected minimax easily - TODO.

For now we also just run out simulations to pick the best move by shuffling the remainder of the deck each time, in a cloned version of the game.

'''
import random
import itertools
import sys
import numpy as np
import torch
import torch.nn as nn
import time
from collections import namedtuple
from copy import deepcopy
from rlcard.agents.dqn_agent_pytorch import DQNAgent, Estimator, Memory
from rlcard.agents.random_agent import RandomAgent

from rlcard.utils.utils import remove_illegal

Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next_state', 'done'])


class DQNAgentEndGame(DQNAgent):
    '''
    Approximate clone of rlcard.agents.dqn_agent.DQNAgent
    that depends on PyTorch instead of Tensorflow
    '''
    def __init__(self,
                 scope,
                 replay_memory_size=20000,
                 replay_memory_init_size=100,
                 update_target_estimator_every=1000,
                 discount_factor=0.99,
                 epsilon_start=1.0,
                 epsilon_end=0.1,
                 epsilon_decay_steps=20000,
                 batch_size=32,
                 action_num=2,
                 state_shape=None,
                 train_every=1,
                 mlp_layers=None,
                 learning_rate=0.00005,
                 device=None,
                 verbose=False,
                 max_neg_reward=-2):

        '''
        Q-Learning algorithm for off-policy TD control using Function Approximation.
        Finds the optimal greedy policy while following an epsilon-greedy policy.

        Args:
            scope (str): The name of the DQN agent
            replay_memory_size (int): Size of the replay memory
            replay_memory_init_size (int): Number of random experiences to sample when initializing
              the reply memory.
            update_target_estimator_every (int): Copy parameters from the Q estimator to the
              target estimator every N steps
            discount_factor (float): Gamma discount factor
            epsilon_start (float): Chance to sample a random action when taking an action.
              Epsilon is decayed over time and this is the start value
            epsilon_end (float): The final minimum value of epsilon after decaying is done
            epsilon_decay_steps (int): Number of steps to decay epsilon over
            batch_size (int): Size of batches to sample from the replay memory
            evaluate_every (int): Evaluate every N steps
            action_num (int): The number of the actions
            state_space (list): The space of the state vector
            train_every (int): Train the network every X steps.
            mlp_layers (list): The layer number and the dimension of each layer in MLP
            learning_rate (float): The learning rate of the DQN agent.
            device (torch.device): whether to use the cpu or gpu
        '''
        self.use_raw = False
        self.scope = scope
        self.replay_memory_init_size = replay_memory_init_size
        self.update_target_estimator_every = update_target_estimator_every
        self.discount_factor = discount_factor
        self.epsilon_decay_steps = epsilon_decay_steps
        self.batch_size = batch_size
        self.action_num = action_num
        self.train_every = train_every
        self.max_neg_reward = max_neg_reward
        self.state_shape = state_shape

        self.verbose = verbose

        # Torch device
        if device is None:
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        # Total timesteps
        self.total_t = 0

        # Total training step
        self.train_t = 0

        # The epsilon decay scheduler
        self.epsilons = np.linspace(epsilon_start, epsilon_end, epsilon_decay_steps)

        # Create estimators
        self.q_estimator = Estimator(action_num=action_num, learning_rate=learning_rate, state_shape=state_shape, \
            mlp_layers=mlp_layers, device=self.device, verbose=verbose)
        self.target_estimator = Estimator(action_num=action_num, learning_rate=learning_rate, state_shape=state_shape, \
            mlp_layers=mlp_layers, device=self.device, verbose=verbose)

        # Create replay memory
        self.memory = Memory(replay_memory_size, batch_size)
        
        # We write (state_hash_key) -> (numeric_best_action, reward) here. The action is Stored
        # in numeric form to keep this in line with the return type from a DQN Agent, but when
        # we query we use the 3 letter versions of the actions as this is how it is set up currently.
        self.end_game_solver = dict()

        self.actions = [['D', 'F', 'F'], ['D', 'F', 'B'], ['D', 'B', 'F'], ['D', 'B', 'B'], ['F', 'D', 'F'], ['F', 'D', 'B'], ['B', 'D', 'F'], ['B', 'D', 'B'], ['F', 'F', 'D'], ['F', 'B', 'D'], ['B', 'F', 'D'], ['B', 'B', 'D']]

        self.added_terms_in_last_thousand = 0

    def step(self, state, env, player_id):
        ''' Predict the action for genrating training data but
            have the predictions disconnected from the computation graph

        Args:
            state (numpy.array): current state

        Returns:
            action (int): an action id
        '''
        # This is where this differs from a normal DQN agent. Here we sample the action from the 
        # the state obs, and pick an action A. If this action A is not in teh legal set, then we 
        # add to the memory buffer a phony transition. (state[obs], A, max_neg_reward, Zero state, Not Done)
        # This is because the next state does not matter, or make sense to have as a transition, 
        # and it does not matter if done.

        # Do some logging.
        if self.total_t % 1000 == 1:
            print("\nSummary of End Game Map: ")
            print(f"Number of the HashMap entries: {len(self.end_game_solver)}")
            print("Size in bytes: {}".format(sys.getsizeof(self.end_game_solver)))
            print("MByes: {}".format(sys.getsizeof(self.end_game_solver) / 1024))
            print(f"Terms added in the last 1000: {self.added_terms_in_last_thousand}")
            self.added_terms_in_last_thousand = 0

        # Check if last deal, and if not then use the normal routine.
        if env.game.dealer.deal_counter == 3:
            
            # First check if there is an entry in the hashmap.
            hash_key = convert_to_hash_key(state.get('obs'))
            hash_map_look_up = self.end_game_solver.get(hash_key) 
            # If so, we can just return this.
            if hash_map_look_up is not None:
                print(f"Using entry from the end game solver map. Key: {hash_key}. Value: {hash_map_look_up}")
                return hash_map_look_up     # This the action.

            self.added_terms_in_last_thousand += 1

            # Otherwise we are at the stage where we need to look up 
            # This would mean we are on the final hand.
            if env.game.game_pointer == env.game.player_to_lead:
                # start_time = time.time()         # Something like 0.15 - 0.35 seconds.
                # TODO: Fix error which is causing problems on the chaning of Oppo Hand. Very strange error.

                # So this is the player who lays first. So there are 5 cards left in the deck, we create a copied game and roll out under the possible 
                # decks these.
                original_game_pointer = deepcopy(env.game.game_pointer)
                copied_game = deepcopy(env.game)
                our_legal_actions = copied_game.get_legal_actions()
                oppo_player = (original_game_pointer + 1) % 2 # As only 2 players.

                # Cards left in the deck and in our oppo hand.
                deck_left = deepcopy(copied_game.dealer.deck + copied_game.players[oppo_player].cards_to_play)

                # We use combinations to get all the possible hands.
                # Need to cast back to lists in order to line up with the type required for the players cards_to_deal hand.
                oppo_possible_hands = list(map(list, itertools.combinations(deck_left, 3)))        

                our_best_action_over_all_shuffle = our_legal_actions[0]
                our_best_reward_over_all_shuffles = -2.0

                for action in our_legal_actions:
                    if self.verbose: print(f"Trying action: {action}")

                    total_reward_for_action = 0
                    new_state, _ = copied_game.step(action)

                    # Loop over every set of three cards possible to draw for opponunun
                    for oppo_hand in oppo_possible_hands:
                        if self.verbose: print(f"Oppo Hand: {oppo_hand}")

                        # Set the cards in oppo hand.
                        copied_game.players[oppo_player].cards_to_play = oppo_hand
                        
                        # We then run through the possible actions our oppo has [which is either 3 or 6] in which case. 
                        oppo_legal_actions = copied_game.get_legal_actions()
                        oppo_best_reward = -2        # Same as our min reward as ZS game.

                        for oppo_action in oppo_legal_actions:
                            if self.verbose: print(f"Trying oppo action: {oppo_action}")
                            # Step forward to game and check reward
                            copied_game.step(oppo_action)
                            oppo_reward = copied_game.get_payoffs()[oppo_player]
                            if oppo_reward > oppo_best_reward:
                                oppo_best_reward = oppo_reward
                                if self.verbose: print(f"Updating Oppo Best Reward: Under action: {oppo_action}, Reward: {oppo_best_reward}")
                            copied_game.step_back()
                        
                        if self.verbose: print(f"Oppo Best Reward: {oppo_best_reward}")
                        
                        # Now we have the best reward that an oppo could have achieved, we can
                        # add our reward to our total using Zero Sum.
                        total_reward_for_action += (-oppo_best_reward)

                    if self.verbose: print(f"After all oppo possible tried under action: {action}")
                    if self.verbose: print(f"Oppo Hands: {oppo_possible_hands}")
                    if self.verbose: print(f"Dealer Deck: {copied_game.dealer.deck}")
                    if self.verbose: print(f"Oppo Cards to Play: {copied_game.players[oppo_player].cards_to_play}")
                    if self.verbose: print(f"Deck Left: {deck_left}")
                    # If we don't seem to reinitalise this, then we seem to have problems. Very strange? Ask about this.
                    oppo_possible_hands = list(map(list, itertools.combinations(deck_left, 3)))
                    if self.verbose: print(f"Reinitatised Oppo Hands: {oppo_possible_hands}")
                    # Once we have the total reward over all oppo hands we can divide for avg.
                    avg_reward_for_action = total_reward_for_action / len(oppo_possible_hands)
                    if self.verbose: print(f"Under Action: {action}, Total Reward: {total_reward_for_action}, Avg Reward: {avg_reward_for_action}")


                    # If the avg reward achieved with this action beats our previous best, then we update.
                    if avg_reward_for_action > our_best_reward_over_all_shuffles:
                        our_best_reward_over_all_shuffles = avg_reward_for_action
                        our_best_action_over_all_shuffle = action

                    # Step back to try the next action.
                    copied_game.step_back()

                if self.verbose: print(f"Best Action: {our_best_action_over_all_shuffle}. Best Reward: {our_best_reward_over_all_shuffles}")

                # Now we can add this to the hash_map under the key for the state.
                hash_key = convert_to_hash_key(state.get('obs'))
                numeric_best_action = self.actions.index(our_best_action_over_all_shuffle)
                print(f"\nPlacing into hashmap. Key: {hash_key}, (Action, Reward): {(numeric_best_action, our_best_reward_over_all_shuffles)}")
                # self.end_game_solver[hash_key] = (numeric_best_action, our_best_reward_over_all_shuffles)
                self.end_game_solver[hash_key] = numeric_best_action
                # print(f"Time taken to complete form 1 move out: {time.time() - start_time}")
                return numeric_best_action
            else:
                # Here we know all of our opponents cards, and all of ours that we will be dealt. We can then copy the game, and check each action returning the best.
                # start_time = time.time()        # Nearly instant: 0.01 seconds.
                original_game_pointer = env.game.game_pointer
                copied_game = deepcopy(env.game)
                legal_actions = copied_game.get_legal_actions()

                best_reward = -2.0
                best_action = legal_actions[0]   # Must always be at least one legal action.
                if self.verbose: print(f"Legal Actions: {legal_actions}")
                for action in legal_actions:
                    # numeric_action = self.actions.index(action)
                    if self.verbose: print(f"Action: {action}")
                    # Step forwards
                    new_state, _ = copied_game.step(action)
                    # TODO: Fix issue with the playerXhands being a bit, and seeming to copy over.
                    reward = copied_game.get_payoffs()[original_game_pointer]
                    if self.verbose: print(f"New State: {new_state}. Payoff: {reward}")
                    if reward > best_reward:
                        best_action = action
                        best_reward = reward
                    # Step back
                    copied_game.step_back()

                if self.verbose: print(f"\nBest Action: {best_action}")
                # Place into the hashmap. Using the convertation to a string whch we normally use in the UCB algorithm.
                hash_key = convert_to_hash_key(state.get('obs'))
                numeric_best_action = self.actions.index(best_action)
                print(f"\nPlacing into hashmap. Key: {hash_key}, (Action, Reward): {(numeric_best_action, best_reward)}")
                # self.end_game_solver[hash_key] = (numeric_best_action, best_reward)
                self.end_game_solver[hash_key] = numeric_best_action
                # print(f"Time taken to complete form 0.5 move out: {time.time() - start_time}")
                return numeric_best_action

           
    
        # If we haven't returned, then we are in the normal part of the game (i.e: Non - end game solving) so we can just proceed as normal.
        A = self.predict(state['obs'])
        potential_action_based_on_q_value = np.random.choice(np.arange(len(A)), p=A)

        # Do the usual negative thing.
        if potential_action_based_on_q_value not in state['legal_actions']:
            # Here we add the extra transition modeling the bad transition.
            current_state = state['obs']
            next_state = np.zeros(self.state_shape)
            reward = self.max_neg_reward
            done = True     # To ensure no value given for discounted terms.
            self.memory.save(current_state, potential_action_based_on_q_value, reward, next_state, done)

        # Then we carry on as usual to remove the illegal move to allow play to roll forward as expected.
        A = remove_illegal(A, state['legal_actions'])
        # print("Final A values: {}".format(A))
        action = np.random.choice(np.arange(len(A)), p=A)
        return action

    def eval_step(self, state):
        ''' Predict the action for evaluation purpose.

        Args:
            state (numpy.array): current state

        Returns:
            action (int): an action id
        '''

        hash_key = convert_to_hash_key(state.get('obs'))
        hash_map_look_up = self.end_game_solver.get(hash_key) 
        # If so, we can just return this.
        if hash_map_look_up is not None:
            # print(f"Using entry from the end game solver map. Key: {hash_key}. Value: {hash_map_look_up}")
            return hash_map_look_up, None

        q_values = self.q_estimator.predict_nograd(np.expand_dims(state['obs'], 0))[0]
        probs = remove_illegal(np.exp(q_values), state['legal_actions'])
        best_action = np.argmax(probs)
        return best_action, probs

    def predict(self, state):
        ''' Predict the action probabilities but have them
            disconnected from the computation graph

        Args:
            state (numpy.array): current state

        Returns:
            q_values (numpy.array): a 1-d array where each entry represents a Q value
        '''
        epsilon = self.epsilons[min(self.total_t, self.epsilon_decay_steps-1)]
        # print("Eps: {}".format(epsilon))
        A = np.ones(self.action_num, dtype=float) * epsilon / self.action_num
        q_values = self.q_estimator.predict_nograd(np.expand_dims(state, 0))[0]
        # print("Q values: {}".format(q_values))
        best_action = np.argmax(q_values)
        A[best_action] += (1.0 - epsilon)
        # print("A values: {}".format(A))

        if self.total_t % 1000 == 1:
            # This is the for the total timesteps so every 1000 steps we can look at the q values:
            print("{}th Iteration".format(self.total_t))
            print("Natural Q Values: {}".format(q_values))
        return A
    
    def raw_q_values(self, state):
        
        # Set eps to be 0.1 to see what the end of the training phase is seeing.
        epsilon = 0.1
        A = np.ones(self.action_num, dtype=float) * epsilon / self.action_num

        q_values = self.q_estimator.predict_nograd(np.expand_dims(state.get('obs'), 0))[0]

        best_action = np.argmax(q_values)
        A[best_action] += (1.0 - epsilon)
        A = remove_illegal(A, state.get('legal_actions')) 
        
        return q_values, A

    def train(self):
        ''' Train the network

        Returns:
            loss (float): The loss of the current batch.
        '''
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.memory.sample()
        if self.verbose: print("State Batch First: {}".format(state_batch[0]))
        if self.verbose: print("Action Batch First: {}".format(action_batch[0]))
        if self.verbose: print("Reward Batch First: {}".format(reward_batch[0]))

        # Calculate best next actions using Q-network (Double DQN)
        q_values_next = self.q_estimator.predict_nograd(next_state_batch)
        best_actions = np.argmax(q_values_next, axis=1)
        if self.verbose: print("Q Values for next from predict: {}".format(q_values_next[0]))
        if self.verbose: print("Best actions: {}".format(best_actions[0]))

        # Evaluate best next actions using Target-network (Double DQN)
        q_values_next_target = self.target_estimator.predict_nograd(next_state_batch)
        if self.verbose: print("Q Values for next from target: {}".format(q_values_next_target[0]))
        target_batch = reward_batch + np.invert(done_batch).astype(np.float32) * \
            self.discount_factor * q_values_next_target[np.arange(self.batch_size), best_actions]
        if self.verbose: print("Target Batch First: {}".format(target_batch[0]))

        # Perform gradient descent update
        state_batch = np.array(state_batch)

        if self.verbose: print("Passing to the network: ")
        if self.verbose: print("State Batch: {}".format(state_batch))
        if self.verbose: print("Action Batch: {}".format(action_batch))
        if self.verbose: print("Target Batch: {}".format(target_batch))

        loss = self.q_estimator.update(state_batch, action_batch, target_batch)
        print('\rINFO - Agent {}, step {}, rl-loss: {}'.format(self.scope, self.total_t, loss), end='')
        if self.verbose: print("\n\n\n")

        # Update the target estimator
        if self.train_t % self.update_target_estimator_every == 0:
            self.target_estimator = deepcopy(self.q_estimator)
            print("\nINFO - Copied model parameters to target network.")

        self.train_t += 1


def convert_to_hash_key(boolean_vector):
    ''' 
    boolean_vector: Should be a numpy vector of shape [1, 108].
    return type: string which represents these as an interger.
    '''
    boolean_vector = np.reshape(boolean_vector, (18, 6))
    # boolean_vector = boolean_vector[0:9]
    integer_representation = [np.where(r==1)[0][0] for r in boolean_vector]
    # Normally just use the string repsentation but will try a smaller just joining sinle digits to see if this is smaller in terms of memory footprint.
    # return repr(integer_representation)
    return ''.join([str(x) for x in integer_representation])
