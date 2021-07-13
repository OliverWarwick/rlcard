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

import random
import numpy as np
import torch
import torch.nn as nn
from collections import namedtuple
from copy import deepcopy
from rlcard.agents.dqn_agent_pytorch import DQNAgent, Estimator, Memory

from rlcard.utils.utils import remove_illegal

Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next_state', 'done'])


class DQNAgentNeg(DQNAgent):
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

    def step(self, state):
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

        A = self.predict(state['obs'])
        potential_action_based_on_q_value = np.random.choice(np.arange(len(A)), p=A)
        # print(f"A: {A}")
        # print(f"Pot action: {potential_action_based_on_q_value}")
        # print(f"Legal states: {state['legal_actions']}")
        # # Is A in the legal moves? 
        if (potential_action_based_on_q_value not in state['legal_actions']):
            # Here we add the extra transition.
            # print(f"Neg transition being added: \nAction: {A}. Legal Actions: {state['legal_actions']}")
            current_state = state['obs']
            next_state = np.zeros(self.state_shape)
            reward = self.max_neg_reward
            done = True     # To ensure no value given for discounted terms.
            self.memory.save(current_state, potential_action_based_on_q_value, reward, next_state, done)

            # if(self.total_t % 250 == 0):
            #     count_of_bad_transitions = 0           
            #     for mem in self.memory.memory:
            #         if np.array_equal(mem[3], np.zeros(self.state_shape)):
            #             count_of_bad_transitions += 1
                    
            #     # print("Mems with neg reward because of bad transition")
            #     print(count_of_bad_transitions)
            #     prop_bad = round(100 * count_of_bad_transitions /  len(self.memory.memory), 5)
            #     print(f"Prop of bad transitions: {prop_bad}")

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
