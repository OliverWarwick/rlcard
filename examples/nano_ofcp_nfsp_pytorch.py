''' An example of learning a NFSP Agent on Leduc Holdem
'''
import os
import torch

import rlcard
from rlcard.agents import NFSPAgentPytorch as NFSPAgent
from rlcard.agents import RandomAgent
from rlcard.utils import set_global_seed, tournament
from rlcard.utils import Logger
from examples.nano_ofcp_q_value_approx import eval_q_value_approx


def training_run(evaluate_every = 1000, 
                evaluate_num = 2500, 
                episode_num = 25000, 
                log_dir = None,
                save_dir = None,
                random_seed = 0):

    # Make environment
    env = rlcard.make('nano_ofcp', config={'seed': 0})
    eval_env_0 = rlcard.make('nano_ofcp', config={'seed': 0})
    eval_env_1 = rlcard.make('nano_ofcp', config={'seed': 0})

    # The intial memory size
    memory_init_size = 1000

    # Train the agent every X steps
    train_every = 16

    # The paths for saving the logs and learning curves
    if log_dir is None:
        log_dir = '.ow_model/experiments/nano_ofcp_nfsp_result_exper/'
    if not os.path.exists(log_dir + '/best_response/player0/'):
        os.makedirs(log_dir + '/best_response/player0/')
    if not os.path.exists(log_dir + '/best_response/player1/'):
        os.makedirs(log_dir + '/best_response/player1/')
    if not os.path.exists(log_dir + '/avg_policy/player0/'):
        os.makedirs(log_dir + '/avg_policy/player0/')
    if not os.path.exists(log_dir + '/avg_policy/player1/'):
        os.makedirs(log_dir + '/avg_policy/player1/')

    # Set up the model saving folder.
    if save_dir is None:
        save_dir = '.ow_model/models/nano_ofcp_nfsp_result_exper/'
    if not os.path.exists(save_dir + '/best_response'):
        os.makedirs(save_dir + '/best_response')
    if not os.path.exists(save_dir + '/avg_policy'):
        os.makedirs(save_dir + '/avg_policy')

    # Set a global seed
    set_global_seed(random_seed)

    # Set agents
    agents = []
    for i in range(env.player_num):
        agent = NFSPAgent(scope='nfsp' + str(i),
                        action_num=env.action_num,
                        state_shape=env.state_shape,
                        hidden_layers_sizes=[64, 64],
                        anticipatory_param=0.25,
                        min_buffer_size_to_learn=memory_init_size,
                        q_replay_memory_init_size=memory_init_size,
                        train_every=train_every,
                        q_train_every=train_every,
                        q_mlp_layers=[64, 64],
                        q_discount_factor=1.0,
                        q_epsilon_start=0.5,
                        device=torch.device('cpu'),
                        rl_learning_rate=0.00005,
                        batch_size=32,
                        evaluate_with='best_response')
        agents.append(agent)
    random_agent = RandomAgent(action_num=eval_env_0.action_num)

    env.set_agents(agents)
    eval_env_0.set_agents([agents[0], random_agent])
    eval_env_1.set_agents([agents[1], random_agent])

    # Init a Logger to plot the learning curve
    logger_best_response_p0 = Logger(log_dir + 'best_response/player0/', csv_name='nfsp_br.csv')
    logger_best_response_p1 = Logger(log_dir + 'best_response/player1/', csv_name='nfsp_br.csv')
    logger_avg_policy_p0 = Logger(log_dir + 'avg_policy/player0/', csv_name='nfsp_ap.csv')
    logger_avg_policy_p1 = Logger(log_dir + 'avg_policy/player1/', csv_name='nfsp_ap.csv')

    best_score_br = 0
    best_score_avg_pol = 0

    for episode in range(episode_num):

        # First sample a policy for the episode
        for agent in agents:
            agent.sample_episode_policy()

        # Generate data from the environment
        trajectories, _ = env.run(is_training=True)

        # Feed transitions into agent memory, and train the agent
        for i in range(env.player_num):
            for ts in trajectories[i]:
                agents[i].feed(ts)

        # Evaluate the performance. Play with random agents.
        if episode % evaluate_every == 0:
        
            tour_score_player_0_br = tournament(eval_env_0, evaluate_num)[0]
            tour_score_player_1_br = tournament(eval_env_1, evaluate_num)[0]
            agents[0].evaluate_with = 'average_policy'
            agents[1].evaluate_with = 'average_policy'
            tour_score_player_0_ap = tournament(eval_env_0, evaluate_num)[0]
            tour_score_player_1_ap = tournament(eval_env_1, evaluate_num)[0]
            agents[0].evaluate_with = 'best_response'
            agents[1].evaluate_with = 'best_response'

            if tour_score_player_0_br > best_score_br:

                state_dict = {}
                for agent in agents:
                    state_dict.update(agent.get_state_dict())
                torch.save(state_dict, os.path.join(save_dir, 'best_response/best_model.pth'))
                best_score_br = tour_score_player_0_br
                print("New highest reward: Best Response")
                print(str(env.timestep) + "  Saving best model. Score vs Random Agent: " + str(best_score_br))
            
            if tour_score_player_0_ap > best_score_avg_pol:

                state_dict = {}
                for agent in agents:
                    state_dict.update(agent.get_state_dict())
                torch.save(state_dict, os.path.join(save_dir, 'avg_policy/best_model.pth'))
                best_score_avg_pol = tour_score_player_0_ap
                print("New highest reward: Avg Policy")
                print(str(env.timestep) + "  Saving best model. Score vs Random Agent: " + str(best_score_avg_pol))


            logger_best_response_p0.log_performance(env.timestep, tour_score_player_0_br)
            logger_avg_policy_p0.log_performance(env.timestep, tour_score_player_0_ap)
            logger_best_response_p1.log_performance(env.timestep, tour_score_player_1_br)
            logger_avg_policy_p1.log_performance(env.timestep, tour_score_player_1_ap)

    # Close files in the logger
    logger_best_response_p0.close_files()
    logger_best_response_p1.close_files()
    logger_avg_policy_p0.close_files()
    logger_avg_policy_p1.close_files()

    # # Plot the learning curve
    # logger.quad_plot('NFSP_BR_P0', 'NFSP_AVG_P0', 'NFSP_BR_P1', 'NFSP_AVG_P1')

    # Save model
    state_dict = {}
    for agent in agents:
        state_dict.update(agent.get_state_dict())
    torch.save(state_dict, os.path.join(save_dir, 'model.pth'))

    agents[0].evaluate_with = 'best_response'
    agents[1].evaluate_with = 'best_response'
    for i, log in enumerate(['best_response/player0/', 'best_response/player1/']):
        q_log_dir = log_dir + log + 'q_values_logs/'
        mean_q_value_diffs = eval_q_value_approx(agents[i], random_agent, sample_size=20, num_rollouts=100, log_dir=q_log_dir)


if __name__ == '__main__':
    for i in range(0,2):
        training_run(
            log_dir = f".ow_model/experiments/nano_ofcp_nfsp_result_exper/run{i}/logs/", 
            save_dir = f".ow_model/experiments/nano_ofcp_nfsp_result_exper/run{i}/model/", 
            random_seed=i*100
        )
