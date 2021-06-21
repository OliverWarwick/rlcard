import os

import rlcard
import torch
import statistics
import numpy as np
import time

from rlcard.agents import DQNAgentPytorch as DQNAgent
from rlcard.agents import RandomAgent
from rlcard.utils import set_global_seed, tournament
from rlcard.utils import Logger



def eval_q_value_approx(agent_1, agent_2, sample_size, num_rollouts):

    """
    Given a set of agents we create an enviroment and simulate the agents strategy over a series of 
    iterations in order to assess how close to the Q value our approximation is under these policies.

    Params:
    agent_1: DQNAgent. This should have the field qnet. i.e: something we can call forwards on to obtain the value from a state
    agent_2: Agent. This just requires a step funtion to get the next move.
    sample_size: Int. Number of games we want to run this over. 
    num_rollouts: Int. How many rollouts should we use per possition under the policies.
    
    Returns:


    """

    # Create a new ofcp env.
    env = rlcard.make('nano_ofcp', config={'seed': 0, 'allow_step_back': True})
    env.set_agents([agent_1, agent_2])

    # Create a logger
    log_dir = '.ow_model/experiments/nano_ofcp_dqn_q_approx/'
    logger = Logger(log_dir)

    # Need to set the global seed in order for this to be the same experiment each time.
    set_global_seed(0)
    q_values_diffs = []

    # TODO: Speed this up through multi processesing, very independent tasks.
    # Currently 206 seconds for (100, 100) run.


    for episode in range(sample_size):

        # Generate data from the environment this function returns a list with 3 elements which is the q value differences over the roll outs done. 
        q_values_diffs.append(roll_out_single_game(env, num_rollouts))

    mean_q_value_differences = np.mean(np.array(q_values_diffs), axis=0)
    print(mean_q_value_differences)

    # logger.log(str(mean_q_value_differences))







def roll_out_single_game(env, num_rollouts):

    # trajectories = [[] for _ in range(env.player_num)]
    # This tracks this for each hand.  
    
    state, player_id = env.reset()
    q_agent_move_number = 0
    q_value_difference = []

    while not env.is_over():

        if player_id == 0:
            # print("\n\n\n\n\nQ agent move counter: {}".format(q_agent_move_number))
            # Get the raw q values and then also the action from the agent, this action will not change so we can save for later.
            raw_q_values, _ = env.agents[player_id].raw_q_values(state)
            original_action, _ = env.agents[player_id].eval_step(state)
            original_q_value = raw_q_values[original_action]
            # print("Original Q Value: {}".format(original_q_value))
            # print("Original State {}, Orginal Action: {}".format(state, original_action))
           
            # Here we can step back and forwards through the future states for the number of roll outs as we know each time we will see a different roll out because of the random seed changing.
            avg_reward = 0.0
            for iteration in range(num_rollouts):
                env.game.dealer.shuffle()
                depth = 0
                while not env.is_over():
                    # print("State before action: ".format(state))
                    action_under_policy, _ = env.agents[player_id].eval_step(state)
                    # TODO: Check this reassignement works, not causing issues.
                    state, player_id = env.step(action_under_policy)
                    # print("Action: {}, State after action: {}".format(action_under_policy, state))
                    # print("Depth: {}".format(depth))
                    depth += 1
                    
                
                # Get the payoff for our players, and add this to the avg reward. We can use the agent as player 1 becuase they will alternate through this.
                payoff = env.get_payoffs()[0]
                # print("Payoff: {}".format(payoff))
                avg_reward = avg_reward + (payoff - avg_reward) / (iteration + 1)
                # print("Finished iteration of roll out: {}, avg score: {}".format(iteration, avg_reward))

                for _ in range(depth):
                    state, player_id = env.step_back()
                    # print("Rollout state: {}".format(state))

            # print("Original State: {}. Original Action: {}. Avg Reward: {}".format(state, original_action, avg_reward))
            # print("Avg Reward: {}, Q Value: {}".format(avg_reward, original_q_value))
            # print("Raw Q Values: {}".format(raw_q_values))
            diff_in_q_value = avg_reward - original_q_value
            # print("Diff in Q value: {}".format(diff_in_q_value))
            
            q_value_difference.append(diff_in_q_value)
            q_agent_move_number += 1
        else:
            original_action, _ = env.agents[player_id].eval_step(state)
        
        # Check that they are the same as before.
        # print(env.get_perfect_information())
        
        # As we're making a move above, we need to check we haven't readed the end of the same.
        state, player_id = env.step(original_action, env.agents[player_id].use_raw)

    return q_value_difference

if __name__ == "__main__":

    env = rlcard.make('nano_ofcp', config={'seed': 0})
    agent_1 = DQNAgent(scope='dqn',
                        action_num=env.action_num,
                        state_shape=env.state_shape,
                        mlp_layers=[128, 128],
                        device=torch.device('cpu'), 
                        epsilon_start = 0.0,
                        epsilon_end = 0.0,
                        epsilon_decay_steps= 1
                        )
    agent_2 = RandomAgent(action_num=env.action_num)
    # agent_2 = DQNAgent(scope='dqn',
    #                     action_num=env.action_num,
    #                     state_shape=env.state_shape,
    #                     mlp_layers=[128, 128],
    #                     device=torch.device('cpu'), 
    #                     epsilon_start = 0.0,
    #                     epsilon_end = 0.0,
    #                     epsilon_decay_steps= 1
    #                     )

    checkpoint = torch.load('.ow_model/models/nano_ofcp_dqn_result_exper/best_model.pth')
    agent_1.load(checkpoint)
    # agent_2.load(checkpoint)

    for x in [10, 100, 1000]:
        for y in [10, 100, 1000]:

            start_time = time.time()
            eval_q_value_approx(agent_1, agent_2, sample_size=x, num_rollouts=y)
            print("Sample Size: {}, Roll Outs: {}, Time: {}".format(x, y, time.time() - start_time))

    # eval_q_value_approx(agent_1, agent_2, sample_size=x, num_rollouts=y)
    




