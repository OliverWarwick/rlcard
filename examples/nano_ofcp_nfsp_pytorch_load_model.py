''' An example of loading pre-trained NFSP model on Leduc Holdem
'''
import os
import torch
import json
import rlcard

from rlcard.agents.nfsp_agent_pytorch import NFSPAgent
from rlcard.agents.random_agent import RandomAgent
from rlcard.utils.utils import set_global_seed, tournament
from rlcard.utils import Logger
from examples.nano_ofcp_q_value_approx import eval_q_value_approx

def env_load_nfsp_agents(agent_kwargs, path):

    # Make environment
    env = rlcard.make('nano_ofcp', config={'seed': 0})

    # Set a global seed
    set_global_seed(0)

    # Load pretrained model
    nfsp_agents = []
    for i in range(env.player_num):
        agent = NFSPAgent(**agent_kwargs)
        nfsp_agents.append(agent)

    # We have a pretrained model here. Change the path for your model.
    checkpoint = torch.load(path)
    for agent in nfsp_agents:
        agent.load(checkpoint)

    env.set_agents([nfsp_agents[0], nfsp_agents[1]])
    return env


def load_env(directory, trainable, mode): 

    agent_path = directory +  "model/final_model.pth"
    json_kwargs_path = directory + "model/agent_kwargs.json"

    json_file = open(json_kwargs_path, "r")
    nfsp_kwargs = json.load(json_file)

    if trainable:
        greedy_parameters = {
            'q_epsilon_start': 0.25,
            'q_epsilon_end': 0.05,
            'q_epsilon_decay_steps': 1,
        }
    else:
        greedy_parameters = {
            'q_epsilon_start': 0.0,
            'q_epsilon_end': 0.0,
            'q_epsilon_decay_steps': 1,
        }

    nfsp_kwargs.update(greedy_parameters)

    return env_load_nfsp_agents(
        nfsp_kwargs,
        agent_path
    )


def training_run(env, log_dir, 
    save_dir, 
    evaluate_every,
    evaluate_num,
    episode_num,
    random_seed):

    # Make environment
    eval_env_0 = rlcard.make('nano_ofcp', config={'seed': random_seed})
    eval_env_1 = rlcard.make('nano_ofcp', config={'seed': random_seed})

    # The paths for saving the logs and learning curves
    if not os.path.exists(log_dir + '/best_response/player0/'):
        os.makedirs(log_dir + '/best_response/player0/')
    if not os.path.exists(log_dir + '/best_response/player1/'):
        os.makedirs(log_dir + '/best_response/player1/')
    if not os.path.exists(log_dir + '/avg_policy/player0/'):
        os.makedirs(log_dir + '/avg_policy/player0/')
    if not os.path.exists(log_dir + '/avg_policy/player1/'):
        os.makedirs(log_dir + '/avg_policy/player1/')

    # Set up the model saving folder.
    if not os.path.exists(save_dir + '/best_response_best'):
        os.makedirs(save_dir + '/best_response_best')
    if not os.path.exists(save_dir + '/avg_policy_best'):
        os.makedirs(save_dir + '/avg_policy_best')

    # Set a global seed
    set_global_seed(random_seed)

    random_agent = RandomAgent(action_num=env.action_num)
    eval_env_0.set_agents([env.agents[0], random_agent])
    eval_env_1.set_agents([env.agents[1], random_agent])

    # Init a Logger to plot the learning curve
    logger_best_response_p0 = Logger(log_dir + 'best_response/player0/', csv_name='nfsp_br.csv')
    logger_best_response_p1 = Logger(log_dir + 'best_response/player1/', csv_name='nfsp_br.csv')
    logger_avg_policy_p0 = Logger(log_dir + 'avg_policy/player0/', csv_name='nfsp_ap.csv')
    logger_avg_policy_p1 = Logger(log_dir + 'avg_policy/player1/', csv_name='nfsp_ap.csv')
    logger = Logger(log_dir)

    best_score_br = 0
    best_score_avg_pol = 0

    for episode in range(episode_num):

        # First sample a policy for the episode
        for agent in env.agents:
            agent.sample_episode_policy()

        # Generate data from the environment
        trajectories, _ = env.run(is_training=True)

        # Feed transitions into agent memory, and train the agent
        for i in range(env.player_num):
            for ts in trajectories[i]:
                env.agents[i].feed(ts)

        # Evaluate the performance. Play with random agents.
        if episode % evaluate_every == 0:
        
            tour_score_player_0_br = tournament(eval_env_0, evaluate_num)[0]
            tour_score_player_1_br = tournament(eval_env_1, evaluate_num)[0]
            env.agents[0].evaluate_with = 'average_policy'
            env.agents[1].evaluate_with = 'average_policy'
            tour_score_player_0_ap = tournament(eval_env_0, evaluate_num)[0]
            tour_score_player_1_ap = tournament(eval_env_1, evaluate_num)[0]
            env.agents[0].evaluate_with = 'best_response'
            env.agents[1].evaluate_with = 'best_response'

            if tour_score_player_0_br > best_score_br:

                state_dict = {}
                for agent in env.agents:
                    state_dict.update(agent.get_state_dict())
                torch.save(state_dict, os.path.join(save_dir, 'best_response_best/model.pth'))
                best_score_br = tour_score_player_0_br
                print("New highest reward: Best Response")
                print(str(env.timestep) + "  Saving best model. Score vs Random Agent: " + str(best_score_br))
            
            if tour_score_player_0_ap > best_score_avg_pol:

                state_dict = {}
                for agent in env.agents:
                    state_dict.update(agent.get_state_dict())
                torch.save(state_dict, os.path.join(save_dir, 'avg_policy_best/model.pth'))
                best_score_avg_pol = tour_score_player_0_ap
                print("New highest reward: Avg Policy")
                print(str(env.timestep) + "  Saving best model. Score vs Random Agent: " + str(best_score_avg_pol))

            logger_best_response_p0.log_performance(env.timestep, tour_score_player_0_br)
            logger_avg_policy_p0.log_performance(env.timestep, tour_score_player_0_ap)
            logger_best_response_p1.log_performance(env.timestep, tour_score_player_1_br)
            logger_avg_policy_p1.log_performance(env.timestep, tour_score_player_1_ap)

            # Log the number of elements in the buffer memories as well.
            # logger.log(f"Number of elements in the player 0 reservoir_buffer [SL]: {len(agents[0]._reservoir_buffer)}")
            # logger.log(f"Number of elements in the player 1 reservoir_buffer [SL]: {len(agents[1]._reservoir_buffer)}")
            # logger.log(f"Number of elements in memory for player 0 [RL]: {len(agents[0]._rl_agent.memory)}")
            # logger.log(f"Number of elements in memory for player 1 [RL]: {len(agents[1]._rl_agent.memory)}")

    # Close files in the logger
    logger_best_response_p0.close_files()
    logger_best_response_p1.close_files()
    logger_avg_policy_p0.close_files()
    logger_avg_policy_p1.close_files()

    # # Plot the learning curve
    # logger.quad_plot('NFSP_BR_P0', 'NFSP_AVG_P0', 'NFSP_BR_P1', 'NFSP_AVG_P1')

    # Save model
    state_dict = {}
    for agent in env.agents:
        state_dict.update(agent.get_state_dict())
    torch.save(state_dict, os.path.join(save_dir, 'final_model.pth'))

    env.agents[0].evaluate_with = 'best_response'
    env.agents[1].evaluate_with = 'best_response'
    for i, log in enumerate(['best_response/player0/', 'best_response/player1/']):
        q_log_dir = log_dir + log + 'q_values_logs/'
        mean_q_value_diffs = eval_q_value_approx(env.agents[i], random_agent, sample_size=25, num_rollouts=25, log_dir=q_log_dir)


def continue_training(directory):

    # We need to form the log and save directory
    log_dir = directory + "logs_cycle_2"
    save_dir = directory + "model_cycle_2"

    run_kwargs = {
        'evaluate_every': 250, 
        'evaluate_num': 500, 
        'episode_num': 500, 
        'random_seed': 0
    }

    env = load_env(directory, trainable=True, mode='best_response')
    training_run(env, log_dir, save_dir, **run_kwargs)




# if __name__ == '__main__':

#     # mode = {'best_response', 'avg_policy'}
#     env = load_env(directory="ow_model/experiments/nano_ofcp_nfsp_saving/run0", trainable=False, mode='best_response')

#     # Evaluate the performance. Play with random agents.
    
#     random_agent = RandomAgent(env.action_num)

#     eval_env_0 = rlcard.make('nano_ofcp', config={'seed': 0})
#     eval_env_0.set_agents([env.agents[0], random_agent])
#     eval_env_1 = rlcard.make('nano_ofcp', config={'seed': 0})
#     eval_env_1.set_agents([env.agents[1], random_agent])

#     evaluate_num = 100
#     reward_0 = tournament(eval_env_0, evaluate_num)[0]
#     reward_1 = tournament(eval_env_1, evaluate_num)[0]
#     print('Reward_0 against random agent: ', reward_0)
#     print('Reward_1 against random agent: ', reward_1)

if __name__ == '__main__':
    directory = "ow_model/experiments/nano_ofcp_nfsp_saving/run0/"
    continue_training(directory)