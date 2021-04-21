''' 
This uses tthe pretrained DQN agent for Nano OFCP to produce some visualisations for how this has learnt.
'''

''' An example of loading pre-trained NFSP model on Leduc Holdem
'''
import os
import torch
import numpy as np
import rlcard
from rlcard.agents.dqn_agent_pytorch import DQNAgent
from rlcard.agents.random_agent import RandomAgent
from rlcard.utils.utils import set_global_seed, tournament

# These are the loading functions we built.
from nano_ofcp_dqn_pytorch_load_model import env_load_dqn_agent_and_random_agent, play_tournament
from rlcard.utils.utils import reorganize, print_card


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

        if isinstance(env.agents[player_id], DQNAgent):
            env.agents[player_id].epsilon_decay_steps = 1
            env.agents[player_id].train_t = 1
            env.agents[player_id].epsilon_end = 0

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
        if isinstance(env.agents[player_id], DQNAgent):

            q_values, with_eps_q_values = env.agents[player_id].raw_q_values(state)

            filtered_action_prob_values = [[env.actions[i], round(q_values[i], 4) if q_values[i] > 0 else round(q_values[i], 3), "", ""] for i in state.get('legal_actions')]
            
            for index, ac in enumerate(state.get('legal_actions')):
                env.step(ac)
                filtered_action_prob_values[index][2] = [[card.rank for card in env.game.players[0].front_row] + [" " for i in range(3 - len(env.game.players[0].front_row))], [card.rank for card in env.game.players[0].back_row] + [" " for i in range(3 - len(env.game.players[0].back_row))]]
                reward = env.get_payoffs()
                filtered_action_prob_values[index][3] = reward.tolist()
                env.step_back()

            print("[    ACTION    | Q_VALUES |     FRONT     |     BACK     |   REWARD   ]| ")
            print(*filtered_action_prob_values, sep='\n')
            
            
            max_index = q_values.tolist().index(max(q_values.tolist()))
            print(max_index)
            print("\nMAX VALUE: ")
            print("   ACTION    | Q VALUE | ALLOWED ")
            print(env.actions[max_index], round(q_values[max_index],4), max_index in state.get('legal_actions'), sep=', ')
            print()

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

    # Final Layout.
    perfect_info = env.get_perfect_information()
    print(perfect_info)

    # print("Payoffs: ")
    # print(payoffs)

    # print("Trajectories:")
    # print(*trajectories, sep='\n')

    return trajectories, payoffs
    


if __name__ == "__main__":
    env = env_load_dqn_agent_and_random_agent(trainable=False, agent_path='/Users/student/rlcard/examples/ow_models/nano_ofcp_dqn_vs_heur/model.pth')

    # Play tournament to see how good the agent is.
    # play_tournament(env, 1000)



    count = 0
    while True:
        play_out_game(env, count)
        play_again = input("\n\nq to quit, or any key to deal another hand: ")
        if play_again == 'q':
            break
        count += 1 