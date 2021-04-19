''' 
An example of learning a Deep-Q Agent on Nano_OFCP.
'''

import tensorflow as tf
import os

import rlcard
from rlcard.agents import RandomAgent, NanoOFCPPerfectInfoAgent
from rlcard.utils import set_global_seed
from rlcard.utils import Logger
import numpy as np
import statistics
import math

def heuristic_agent_tournament(env, num):
    ''' Evaluate he performance of the agents in the environment

    Args:
        env (Env class): The environment to be evaluated.
        num (int): The number of games to play.

    Returns:
        A list of avrage payoffs for each player
    '''
    payoffs = [0 for _ in range(env.player_num)]
    # payoff_list_heur = []
    # print("Payoffs: {}".format(payoffs))
    counter = 0
    while counter < num:
        traj, _payoffs = env.heuristic_agent_run(is_training=False)
        # payoff_list_heur.append(_payoffs[0])
        # print("Traj for player 0: ")
        # print(*traj[0], sep='\n')
        # print("PAYOFF: " + str(payoffs))
        # print("_payoffs from env: {}".format(_payoffs))
        # print("Type of _payoffs: {}".format(type(_payoffs)))
        # print("Payoffs from tournament: {}".format(_payoffs))
        if isinstance(_payoffs, list):
            for _p in _payoffs:
                for i, _ in enumerate(payoffs):
                    payoffs[i] += _p[i]
                counter += 1
        else:
            # If the payoff is not 2 for the first player, look at the result.
            # if _payoffs[0] != 2:
                # print("PAYOFF: " + str(_payoffs))
                # print("TRAJ 0: ")
                # for ts in traj[0]:
                    
                #     print('State: {} \nAction: {} \nReward: {} \nNext State: {} \n\n'.format(ts[0], ts[1], ts[2], ts[3]))
                # print(*traj[0][0], sep='\n')
                # print(*traj[0][1], sep='\n')
                # print(*traj[0][2], sep='\n')
                # print("\n\n\n")
                # print("TRAJ 1: ")
                # print(*traj[1][0], sep='\n')
                # print("\n\n\n")
            for i, _ in enumerate(payoffs):
                payoffs[i] += _payoffs[i]
            counter += 1
    # print("Counter: " + str(counter))
    for i, _ in enumerate(payoffs):
        payoffs[i] /= counter
    # print(payoff_list_heur)
    # sd = np.std(payoff_list_heur, axis=0)
    # print(sd)
    # return payoffs[0], payoffs[1], sd
    return payoffs

if __name__ == '__main__':

    # Make environment
    env = rlcard.make('nano_ofcp', config={'seed': 0})
    eval_env = rlcard.make('nano_ofcp', config={'seed': 0})

    # Set a global seed
    set_global_seed(0)

    results = []

    for alpha in [1.0, 0.75, 0.5, 0.25, 0.1]:

        evaluate_num = 10000

        # Set up the agents
        heuristic_agent = NanoOFCPPerfectInfoAgent(action_num=env.action_num, use_raw=True, alpha=alpha)
        random_agent = RandomAgent(action_num=eval_env.action_num)
        env.set_agents([heuristic_agent, random_agent])
        eval_env.set_agents([heuristic_agent, random_agent])


            # Generate data from the environment
        # trajectories, _ = env.heuristic_agent_run(is_training=False)
        # print(trajectories)
        
        # Evaluate the performance. Play with random agents.
        # if episode % evaluate_every == 0:
        #     print(tournament(eval_env, evaluate_num)[0])
        # print(trajectories)
        # print("Deck before starting: ")
        # print(eval_env.game.dealer.deck)
        score, _, sd = heuristic_agent_tournament(eval_env, evaluate_num)
        print("Alpha: {} Percentage explored: {:.2f} Score: {} S.D: {}".format(alpha, heuristic_agent.explored_percentage * 100, score, sd))
        results.append([alpha, heuristic_agent.explored_percentage * 100, score, score - 1.96 * (sd / math.pow(evaluate_num, 0.5)), score + 1.96 * (sd / math.pow(evaluate_num, 0.5))])

    print(results)

    results_np = np.array(results)
    np.savetxt("frac_sampling_action.csv", results_np, delimiter=",")