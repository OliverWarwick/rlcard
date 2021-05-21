''' 
This uses tthe pretrained DQN agent for Nano OFCP to produce some visualisations for how this has learnt.
'''

''' An example of loading pre-trained NFSP model on Leduc Holdem
'''
import os
import torch
import numpy as np
import rlcard
from rlcard.agents.nfsp_agent_pytorch import NFSPAgent
from rlcard.agents.random_agent import RandomAgent
from rlcard.utils.utils import set_global_seed, tournament

# These are the loading functions we built.
from nano_ofcp_nfsp_pytorch_load_model import load_nfsp_from_memory
from rlcard.utils.utils import reorganize, print_card, tournament
from tabulate import tabulate


def play_out_game(env, count):

    set_global_seed(100)

    trajectories = [[] for _ in range(env.player_num)]
    state, player_id = env.reset()
    if count != 0:
        print("Shuffling")
        env.game.dealer.shuffle()

    # Loop to play the game
    trajectories[player_id].append(state)
    while not env.is_over():

        if isinstance(env.agents[player_id], NFSPAgent):
            # env.agents[player_id].epsilon_decay_steps = 1
            # env.agents[player_id].train_t = 1
            # env.agents[player_id].epsilon_end = 0

        # Display the state as we would for a human playing that decision:
        # Get the ending arrangement.
            rows = state.get('hands')
            print(rows)

            print("\n\nSTARTING NEXT DEAL\n\n")
            print('===============   MY HAND  ===============\n')
            print('===============   Cards to Play      ===============')
            print_card(rows[0])
            print('===============   Front Row       ===============')
            print_card(rows[1])
            print('===============   Back Row        ===============')
            print_card(rows[2])
            print('===============   Discard Pile       ===============')
            print_card(rows[3])

            print('\n===============   OPPO HAND   ===============\n')
            print('===============   Front Row       ===============')
            print_card(rows[4])
            print('===============   Back Row        ===============')
            print_card(rows[5])
            print()

        # Agent plays
        action, probablities = env.agents[player_id].eval_step(state)

        if isinstance(env.agents[player_id], NFSPAgent) and player_id==1:

            # Here if we are playing in the best response mode we make a call to a Q network, so we can just use the raw q values funcitons we normally use. Otherwise, we only can get probablitlies which come from the policy network.
            if env.agents[player_id].evaluate_with == 'best_response':
                q_values, with_eps_q_values = env.agents[player_id]._rl_agent.raw_q_values(state)
                filtered_action_prob_values = [[env.actions[i], round(q_values[i], 4) if q_values[i] > 0 else round(q_values[i], 3), "", ""] for i in state.get('legal_actions')]
                
                for index, ac in enumerate(state.get('legal_actions')):
                    env.step(ac)
                    filtered_action_prob_values[index][2] = [[card.rank for card in env.game.players[player_id].front_row] + [" " for i in range(3 - len(env.game.players[player_id].front_row))], [card.rank for card in env.game.players[player_id].back_row] + [" " for i in range(3 - len(env.game.players[player_id].back_row))]]
                    reward = env.get_payoffs()
                    filtered_action_prob_values[index][3] = reward.tolist()
                    env.step_back()

                print(tabulate(filtered_action_prob_values, headers=['ACTION', 'Q_VALUES', 'HANDS', 'REWARD']))
                max_index = q_values.tolist().index(max(q_values.tolist()))
                print(tabulate([["MAX", env.actions[max_index], round(q_values[max_index],4), max_index in state.get('legal_actions')], ["SAMPLED", env.actions[action], round(q_values[action], 4), action in state.get('legal_actions')]], headers = ['TYPE', 'ACTION', 'Q VALUE', 'ALLOWED']))
                print()

            else:
                # We are in the average policy case now.
                filtered_action_prob_values = [[env.actions[i], round(probablities[i], 4) if probablities[i] > 0 else round(probablities[i], 3), "", ""] for i in state.get('legal_actions')]
                for index, ac in enumerate(state.get('legal_actions')):
                    env.step(ac)
                    filtered_action_prob_values[index][2] = [[card.rank for card in env.game.players[player_id].front_row] + [" " for i in range(3 - len(env.game.players[player_id].front_row))], [card.rank for card in env.game.players[player_id].back_row] + [" " for i in range(3 - len(env.game.players[player_id].back_row))]]
                    reward = env.get_payoffs()
                    filtered_action_prob_values[index][3] = reward.tolist()
                    env.step_back()
                # print("[    ACTION    |   PROB   |     FRONT     |     BACK     |  REWARD  | ")
                # print(*filtered_action_prob_values, sep='\n')
                print(tabulate(filtered_action_prob_values, headers = ['ACTION', 'PROB', 'HANDS', 'REWARD'], tablefmt='github', colalign=("left",)))
                print()
                max_index = probablities.tolist().index(max(probablities.tolist()))
                chosen_elements = [["MAX", env.actions[max_index], round(probablities[max_index],4)], [" SAMPLED ", env.actions[action], round(probablities[action], 4)]]
                print(tabulate(chosen_elements, headers = ['TYPE', 'ACTION', 'PROB'], tablefmt='github', colalign=('left',)))

            

        # Environment steps
        next_state, next_player_id = env.step(action, env.agents[player_id].use_raw)
        # Save action
        trajectories[player_id].append(action)

        # Set the state and player
        state = next_state
        player_id = next_player_id

        # Save state.
        if not env.game.is_over():
            trajectories[player_id].append(state)

    # Add a final state to all the players
    for player_id in range(env.player_num):
        state = env.get_state(player_id)
        trajectories[player_id].append(state)

    # Payoffs
    # print("Final call to payoff\n\n\n\n")
    payoffs = env.get_payoffs()

    # Reorganize the trajectories
    trajectories = reorganize(trajectories, payoffs)
    print(trajectories[0])

    # Final Layout.
    perfect_info = env.get_perfect_information()
    print(perfect_info)

    # print("Payoffs: ")
    # print(payoffs)

    # print("Trajectories:")
    # print(*trajectories, sep='\n')

    return trajectories, payoffs
    


if __name__ == "__main__":
    mode = 'best_response'
    nfsp_agents = load_nfsp_from_memory(check_point_path='/Users/student/rlcard/examples/models/nano_ofcp_nfsp_colabs_run/model.pth', mode=mode)

    # Need to do some fiddling here.
    env = rlcard.make('nano_ofcp', config={'seed': 0, 'allow_step_back':True})
    env.set_agents(nfsp_agents)
    eval_env = rlcard.make('nano_ofcp', config={'seed': 0, 'allow_step_back':True})
    random_agent = RandomAgent(action_num=eval_env.action_num)
    eval_env.set_agents([nfsp_agents[1], random_agent])


    # Play tournament to see how good the agent is.
    # reward = tournament(eval_env, 1000)[0]
    # print('Average reward against random agent: ', reward)

    count = 0
    while True:
        play_out_game(env, count)
        play_again = input("\n\nq to quit, or any key to deal another hand: ")
        if play_again == 'q':
            break
        count += 1 