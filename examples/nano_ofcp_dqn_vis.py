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



def play_out_game(env):

    trajectories = [[] for _ in range(env.player_num)]
    state, player_id = env.reset()

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
            values = env.agents[player_id].raw_q_values(state.get('obs'))
            filtered_action_prob_values = [(env.actions[i], values[i], probablities[i]) for i in state.get('legal_actions')]
            print(*filtered_action_prob_values, sep='\n')

        # print(probablities[state.get('legal_actions')])

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
    payoffs = env.get_payoffs()

    # Reorganize the trajectories
    trajectories = reorganize(trajectories, payoffs)

    # Final Layout.
    perfect_info = env.get_perfect_information()
    print(perfect_info)

    return trajectories, payoffs
    


if __name__ == "__main__":
    env = env_load_dqn_agent_and_random_agent(trainable=False)
    play_out_game(env)
    