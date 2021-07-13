''' An example of learning a Deep-Q Agent on Nano OFCP using a PyTorch implimentation 
'''
import torch
import os
import json 
from copy import deepcopy
import argparse
import rlcard
from rlcard.utils import set_global_seed, tournament
from rlcard.utils import Logger
from examples.nano_ofcp_q_value_approx import eval_q_value_approx
from rlcard.agents import DQNAgentPytorchNeg as DQNAgentNeg, RandomAgent
# from nano_ofcp_dqn_pytorch_load_model import load_dqn_agent


def training_run(agent_kwargs, log_dir, save_dir, evaluate_every, evaluate_num, episode_num, random_seed, random_finish, early_finish, mid_finish, q_value_est_sample_size, q_value_est_num_rollout):

    # Make environment
    training_env_vs_random = rlcard.make('nano_ofcp', config={'seed': random_seed})
    training_env_vs_early_dqn = rlcard.make('nano_ofcp', config={'seed': random_seed})
    training_env_vs_mid_dqn = rlcard.make('nano_ofcp', config={'seed': random_seed})
    # heuristic_agent_env = rlcard.make('nano_ofcp', config={'seed': random_seed})
    eval_env_vs_random = rlcard.make('nano_ofcp', config={'seed': random_seed})

    # The paths for saving the logs and learning curves
    if log_dir is None:
        log_dir = 'ow_model/experiments/nano_ofcp_dqn_league/'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Set up the model saving folder.
    if save_dir is None:
        save_dir = 'ow_model/models/nano_ofcp_dqn_league/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Set a global seed
    set_global_seed(random_seed)

    # agent = load_dqn_agent(agent_kwargs, agent_type, agent_path)

    random_agent = RandomAgent(action_num=eval_env_vs_random.action_num)
    # Update for use with the when we don't want any updates.
    updated_agent_kwargs = deepcopy(agent_kwargs)
    updated_agent_kwargs['learning_rate'] = 0.0
    updated_agent_kwargs['epsilon_start'] = 0.0
    updated_agent_kwargs['epsilon_end'] = 0.0
    main_dqn_agent = DQNAgentNeg(**agent_kwargs)
    # We can initalise a agent, and then load from a check point.
    # We also need to set the update rate to be zero here for our saved agents, as otherwise
    # they will modify themselves which is not allowed, and we cannot switch training on 
    # one for one player.
    early_dqn_agent = DQNAgentNeg(**updated_agent_kwargs)
    mid_dqn_agent = DQNAgentNeg(**updated_agent_kwargs)
    
    training_env_vs_random.set_agents([main_dqn_agent, random_agent])
    eval_env_vs_random.set_agents([main_dqn_agent, random_agent])
    # The final enviroment is filled once we have our agent.

    # Init a Logger to plot the learning curve, and use the name of the model so we can 
    # later plot these.
    logger_vs_random = Logger(log_dir, csv_name="dqn_league_learning_random.csv")
    logger_vs_early = Logger(log_dir, csv_name="dqn_league_learning_early.csv")
    logger_vs_mid = Logger(log_dir, csv_name="dqn_league_learning_mid.csv")

    # Display infomation about the agents networks.
    # print("Agents network shape: {}".format(agent.q_estimator.qnet))
    # print("Agents network layers: {}".format(agent.q_estimator.qnet.fc_layers))

    best_score = 0

    for episode in range(episode_num):

        # If just ran out the last episode training against an agnet, have a check of how close
        # to the Q values we got incase we need to run for longer in order to converge.
        if episode == random_finish:
            # Test against the mid DQN agent as this is the last thing we trained against.
            q_value_log_dir = log_dir + 'q_values_logs/random/'
            if not os.path.exists(q_value_log_dir):
                os.makedirs(q_value_log_dir)
            mean_q_value_diffs = eval_q_value_approx(main_dqn_agent, random_agent, sample_size=q_value_est_sample_size, num_rollouts=q_value_est_num_rollout, log_dir=q_value_log_dir)
        if episode == early_finish:
            q_value_log_dir = log_dir + 'q_values_logs/early/'
            if not os.path.exists(q_value_log_dir):
                os.makedirs(q_value_log_dir)
            mean_q_value_diffs = eval_q_value_approx(main_dqn_agent, early_dqn_agent, sample_size=q_value_est_sample_size, num_rollouts=q_value_est_num_rollout, log_dir=q_value_log_dir)
        if episode == mid_finish:
            q_value_log_dir = log_dir + 'q_values_logs/mid/'
            if not os.path.exists(q_value_log_dir):
                os.makedirs(q_value_log_dir)
            mean_q_value_diffs = eval_q_value_approx(main_dqn_agent, early_dqn_agent, sample_size=q_value_est_sample_size, num_rollouts=q_value_est_num_rollout, log_dir=q_value_log_dir)


        # Make an early and mid version of the agent.
        if episode == int(0.9 * random_finish):
            # TODO: Streamline this, don't really need to save it and work backwards
            # instead we could just overwrite the weights in the networks, but seems messy.
            # TODO: Is there a better way than setting the weights to be zero. 
            print(f"\nSaving for the early DQN agent. Ep Number: {episode}")
            state_dict = main_dqn_agent.get_state_dict()
            torch.save(state_dict, os.path.join(save_dir, 'early_dqn_model.pth'))
            # Load this into our agent.
            checkpoint = torch.load(os.path.join(save_dir, 'early_dqn_model.pth'))
            early_dqn_agent.load(checkpoint)
            training_env_vs_early_dqn.set_agents([main_dqn_agent, early_dqn_agent])
        if episode == int(0.9 * early_finish):
            print(f"\nSaving for the mid DQN agent. Ep Number: {episode}")
            state_dict = main_dqn_agent.get_state_dict()
            torch.save(state_dict, os.path.join(save_dir, 'mid_dqn_model.pth'))
            # Load this into our agent.
            checkpoint = torch.load(os.path.join(save_dir, 'mid_dqn_model.pth'))
            mid_dqn_agent.load(checkpoint)
            training_env_vs_mid_dqn.set_agents([main_dqn_agent, mid_dqn_agent])

        # Depending on how far we are into the game, we either play against a random agent, or 
        # against a weaker version of our agent, which will have been filled from after 5000 epoches.
        if episode < random_finish:
            # Generate data from the environment
            trajectories, _ = training_env_vs_random.run(is_training=True)
        elif episode < early_finish:
            trajectories, _ = training_env_vs_early_dqn.run(is_training=True)
        elif episode < mid_finish:
            trajectories, _ = training_env_vs_mid_dqn.run(is_training=True)
        else:
            # TODO: Figure out what to do here.
            trajectories, _ = training_env_vs_mid_dqn.run(is_training=True)


        # Feed transitions into agent memory, and train the agent
        # TODO: Could just feed in from several targets.
        for ts in trajectories[0]:
            main_dqn_agent.feed(ts)

        # Evaluate the performance. Play with random agents.
        if episode % evaluate_every == 0:
            
            tour_score_vs_random = tournament(eval_env_vs_random, evaluate_num)[0]
            if tour_score_vs_random > best_score:
                state_dict = main_dqn_agent.get_state_dict()
                torch.save(state_dict, os.path.join(save_dir, 'best_model.pth'))
                best_score = tour_score_vs_random
                logger_vs_random.log(str(episode) + "  Saving best model. Expected Reward: " + str(best_score))
            
            print("\nScore vs Random Agent")
            logger_vs_random.log_performance(episode, tour_score_vs_random)

            if episode >= random_finish:
                tour_score_vs_early = tournament(training_env_vs_early_dqn, evaluate_num)[0]
                print("\nScore vs Early Agent")
                logger_vs_early.log_performance(episode - random_finish, tour_score_vs_early)
            if episode >= early_finish:
                print("\nScore vs Mid Agent")
                tour_score_vs_mid = tournament(training_env_vs_mid_dqn, evaluate_num)[0]
                logger_vs_mid.log_performance(episode - early_finish, tour_score_vs_mid)
            

    # Close files in the logger
    logger_vs_random.close_files()
    logger_vs_early.close_files()
    logger_vs_mid.close_files()

    # Plot the learning curve
    logger_vs_random.plot('dqn_vs_randoms')
    logger_vs_early.plot('dqn_vs_early')
    logger_vs_mid.plot('dqn_vs_mid')

    # Save model
    state_dict = main_dqn_agent.get_state_dict()
    torch.save(state_dict, os.path.join(save_dir, 'model.pth'))


if __name__ == '__main__':

    # Parsing for the arguments.
    parser = argparse.ArgumentParser(description='Type of Vis')
    parser.add_argument('--local', dest='run_env_local', type=bool, default=True, help='Whether we are running in colabs or not. True / False')
    args = parser.parse_args()

    if args.run_env_local:
        directory = "ow_model/experiments/nano_ofcp_dqn_league_multi_play/run0/"
    else:
        directory = '/content/drive/MyDrive/msc_thesis/experiments/nano_ofcp_dqn_league_multi_play/'

    trainable = True

    # We need to form the log and save directory
    log_dir = directory + "logs_league"
    save_dir = directory + "logs_league"

    # run_kwargs = {
    #     'evaluate_every': 250, 
    #     'evaluate_num': 50, 
    #     'episode_num': 1000, 
    #     'random_seed': 0,
    #     'random_finish': 300, 
    #     'early_finish': 600, 
    #     'mid_finish': 1000,
    #     'q_value_est_sample_size': 25, # 100 at normal test time.
    #     'q_value_est_num_rollout': 25 # 100 at test time
    # }

    run_kwargs = {
        'evaluate_every': 5000, 
        'evaluate_num': 10000, 
        'episode_num': 250000, 
        'random_seed': 0,
        'random_finish': 75000, 
        'early_finish': 150000, 
        'mid_finish': 250000,
        'q_value_est_sample_size': 100, # 100 at normal test time.
        'q_value_est_num_rollout': 100 # 100 at test time
    }
    
    # agent_path = directory + "model/model.pth"
    # agent_type = "DQN_NEG"

    # json_kwargs_path = directory + "model/agent_kwargs.json"
    # json_file = open(json_kwargs_path, "r")
    # agent_kwargs = json.load(json_file)

    agent_kwargs = {
            'scope': 'dqn',
            'state_shape': 108,
            'action_num': 12,
            'device': 'cpu',
            'replay_memory_init_size': 1000,
            'update_target_estimator_every': 2500,
            'train_every': 1,
            'mlp_layers': [128, 128, 64],
            'learning_rate': 0.00005,
            'batch_size': 64,
            'epsilon_start': 0.3,
            'epsilon_end': 0.05,
            'epsilon_decay_steps': run_kwargs['episode_num'],
            'discount_factor': 1.0,
            'verbose': False,
            'max_neg_reward': -2
        }

    # greedy_parameters = {
    #     'epsilon_start': 0.5,
    #     'epsilon_end': 0.05,
    #     'epsilon_decay_steps': run_kwargs['episode_num'],
    # }

    # agent_kwargs.update(greedy_parameters)
    
    training_run(agent_kwargs, log_dir, save_dir, **run_kwargs)
