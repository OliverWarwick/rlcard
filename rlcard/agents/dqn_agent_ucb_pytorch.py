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
OW Edit: Adding this in order to test out some ideas from the OPIQ paper.
This should just keep track of each state and action pair visit count and use
these during the update process.  
ONLY FOR USE IN SMALL S/A SPACES
We count by having a entry in a hashmap for each state, and an assisoated array 
which is size # fixed actions. 
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


class DQNAgentUCB(DQNAgent):
    '''
    Approximate clone of rlcard.agents.dqn_agent.DQNAgent
    that depends on PyTorch instead of Tensorflow
    '''
    def __init__(self,
                 scope,
                 replay_memory_size=20000,
                 replay_memory_init_size=100,
                 update_target_estimator_every=2500,        
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
                 optimisitic_bias_on_action=4,
                 optimisitic_bias_on_bootstrap=4,
                 optimism_decay=0.5):

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

        # This is a map of states to numpy vectors.
        # The keys are raw_obs from the extract state function to numpy vector.
        self.count_map = dict()
        self.optimisitic_bias_on_action = optimisitic_bias_on_action
        self.optimisitic_bias_on_bootstrap = optimisitic_bias_on_bootstrap
        self.optimism_decay = optimism_decay

    def step(self, state):
        ''' Predict the action for genrating training data but
            have the predictions disconnected from the computation graph

        Args:
            state (numpy.array): current state

        Returns:
            action (int): an action id
        '''
        # Here we want to check to see if we have seen the state before


        # We have to convert the numpy array to a string as numpy array is not hashable, but strings
        # are as they are immutable. 
        # TODO: Can we just use a prefix of the full vector, rather than details in decard piles
        # Does this change if we aren't player 0?
        hash_key = convert_to_hash_key(state.get('obs'))
        state_action_counts = self.count_map.get(hash_key)
        if state_action_counts is None:
            # Need to then create this this, for illegal actions give large positive value. 
            self.count_map[hash_key] = np.array([0 if x in state.get('legal_actions') else 100 for x in range(self.action_num)], dtype=int)


        # Can now call predict as we know count_map must have an entry for this.
        A = self.predict(state['obs'])
        A = remove_illegal(A, state['legal_actions'])
        # We now have an e-greedy profile over the probablities
        action = np.random.choice(np.arange(len(A)), p=A)
        if self.verbose: print("Action number: {}".format(action))

        # Add to the (s,a) pair 1 to show we have visited. Can use the state action as 
        self.count_map.get(hash_key)[action] = self.count_map.get(hash_key)[action] + 1
        return action

    def eval_step(self, state):
        ''' Predict the action for evaluation purpose.

        Args:
            state (numpy.array): current state

        Returns:
            action (int): an action id
        '''
        # hash_key = convert_to_hash_key(state.get('obs'))
        # state_action_counts = self.count_map.get(hash_key)
        # if state_action_counts is None:
        #     # Need to then create this this.
        #     self.count_map[hash_key] = np.zeros(self.action_num, dtype=int)

        natural_q_values = self.q_estimator.predict_nograd(np.expand_dims(state['obs'], 0))[0]
        # ucb_q_values = natural_q_values + np.multiply(self.optimisitic_bias_on_action, np.power((self.count_map[hash_key] + np.ones(self.action_num, dtype=float)), -self.optimism_decay))
        probs = remove_illegal(np.exp(natural_q_values), state['legal_actions'])
        best_action = np.argmax(probs)

        return best_action, probs

    def predict(self, state_obs):
        ''' Predict the action probabilities but have them
            disconnected from the computation graph

        Args:
            state (numpy.array): current state

        Returns:
            q_values (numpy.array): a 1-d array where each entry represents a Q value
        '''
        

        epsilon = self.epsilons[min(self.total_t, self.epsilon_decay_steps-1)]
        A = np.ones(self.action_num, dtype=float) * epsilon / self.action_num

        # Here we are using the optimisitic_bias during action selection.
        hash_key = convert_to_hash_key(state_obs)
        state_action_counts = self.count_map.get(hash_key)
        if state_action_counts is None:
            # Need to then create this this.
            self.count_map[hash_key] = np.zeros(self.action_num, dtype=int)
            raise Exception("Empty look up in the map. - OW. This may happen when we are using eval or predict on new states at test time")

        natural_q_values = self.q_estimator.predict_nograd(np.expand_dims(state_obs, 0))[0]
        ucb_q_values = natural_q_values + np.multiply(self.optimisitic_bias_on_action, np.power((self.count_map[hash_key] + np.ones(self.action_num, dtype=float)), -self.optimism_decay))

        if self.verbose: print("Nat Q values: {}".format(natural_q_values))
        if self.verbose: print("Count map: {}".format(self.count_map[hash_key]))
        if self.verbose: print("UCB Q values: {}".format(ucb_q_values))

        best_action = np.argmax(ucb_q_values)
        A[best_action] += (1.0 - epsilon)

        if self.total_t % 1000 == 1:
            # This is the for the total timesteps so every 1000 steps we can look at the q values:
            print("{}th Iteration".format(self.total_t))
            print("State: {}".format(state_obs))
            print("Hash Key: {}".format(hash_key))
            print("Count Map: {}".format(self.count_map[hash_key]))
            print("Natural Q Values: {}".format(natural_q_values))
            print("Opt Term: {}".format(np.multiply(self.optimisitic_bias_on_action, np.power((self.count_map[hash_key] + np.ones(self.action_num, dtype=float)), -self.optimism_decay))))
            print("UCB Q Values: {}\n\n".format(ucb_q_values))

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

        # We then need to add the optimisitic bias terms onto the rewards.
        # Calculate best next actions using Q-network (Double DQN)

        # OW - To the next state we add the values from the optimisitic bias term. 
        # OW - First we need to check that the next_state_batch exists, and if not add.
        
        # TODO: Vectorise.
        # Look up the next state to see 
        # for next_state in next_state_batch:
            # Check if exists.
            # if self.count_map.get(next_state_batch['raw_obs']) is None:
                # Need to then create this this.
                # self.count_map[next_state_batch['raw_obs']] = np.zeros(self.action_num)
        
        nautral_q_values_next = self.q_estimator.predict_nograd(next_state_batch)
        # Add on the optimistic bias term using the bootstrapping value.
        # state_action_counts = np.array([np.array(self.count_map.get(next_state)) for next_state in next_state_batch])
        # if self.verbose: print("Natural Q Values First: {}".format(natural_q_values_next[0]))
        # if self.verbose: print("state_action_counts First: {}".format(state_action_counts[0]))
        # best_actions = natural_q_values + np.multiply(self.optimisitic_bias_on_bootstrap, np.power((state_action_counts + np.ones((self.batch_size, self.action_num), dtype=float)), -self.optimism_decay))
        best_actions = np.argmax(nautral_q_values_next, axis=1)
        if self.verbose: print("Q Values for next from predict First: {}".format(best_actions[0]))
        if self.verbose: print("Best actions First: {}".format(best_actions[0]))

        # Evaluate best next actions using Target-network (Double DQN)
        natural_q_values_next_target = self.target_estimator.predict_nograd(next_state_batch)
        # ucb_q_values_next_target = natural_q_values_next_target + np.multiply(self.optimisitic_bias_on_bootstrap, np.power((state_action_counts + np.ones((self.batch_size, self.action_num), dtype=float)), -self.optimism_decay))

        # if self.verbose: print("Q Values for next from target: {}".format(q_values_next_target[0]))
        target_batch = reward_batch + np.invert(done_batch).astype(np.float32) * \
            self.discount_factor * natural_q_values_next_target[np.arange(self.batch_size), best_actions]
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


# Function for converting the state to a string which can be used as a hashkey in a 
# map for looking up the values.
def convert_to_hash_key(boolean_vector):
    ''' 
    boolean_vector: Should be a numpy vector of shape [1, 108].
    return type: string which represents these as an interger.
    '''
    boolean_vector = np.reshape(boolean_vector, (18, 6))
    # This would only get our current cards to play, and those which we have laid, hopefully leacing to a smaller number of possible combos.
    boolean_vector = boolean_vector[0:9]
    integer_representation = [np.where(r==1)[0][0] for r in boolean_vector]
    # Normally just use the string repsentation but will try a smaller just joining sinle digits to see if this is smaller in terms of memory footprint.
    # return repr(integer_representation)
    return ''.join([str(x) for x in integer_representation])
