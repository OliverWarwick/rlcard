''' 
An example of learning a Deep-Q Agent on Nano_OFCP.
'''

import torch
import os

import rlcard
from rlcard.agents import DQNAgentPytorch as DQNAgent, NanoOFCPPerfectInfoAgent, RandomAgent
from rlcard.utils import set_global_seed, tournament
from rlcard.utils import Logger
from nano_ofcp_heuristic_vs_random import heuristic_agent_tournament


def training_run(evaluate_every = 500, 
                evaluate_num = 2500, 
                episode_num = 15000, 
                memory_init_size = 2500, 
                train_every = 1, 
                log_dir = None,
                save_dir = None,
                random_seed = 0):

    # Make environment
    env = rlcard.make('nano_ofcp', config={'seed': 0})
    eval_env_random = rlcard.make('nano_ofcp', config={'seed': 0})
    eval_env_heur = rlcard.make('nano_ofcp', config={'seed': 0})

    # The paths for saving the logs and learning curves
    if log_dir is None:
        log_dir = '.ow_model/experiments/nano_ofcp_dqn_and_heur/'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Set up the model saving folder.
    if save_dir is None:
        save_dir = '.ow_model/models/nano_ofcp_dqn_and_heur/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Set a global seed
    set_global_seed(random_seed)

    agent = DQNAgent(scope='dqn',
                    action_num=env.action_num,
                    replay_memory_init_size=memory_init_size,
                    train_every=train_every,
                    state_shape=env.state_shape,
                    mlp_layers=[64, 64],
                    device=torch.device('cpu'),
                    epsilon_decay_steps=episode_num,
                    epsilon_start=0.5,
                    epsilon_end=0.1,
                    learning_rate=0.00005,
                    update_target_estimator_every=evaluate_every,
                    verbose=False, 
                    batch_size=32,
                    discount_factor=1.0)

    random_agent = RandomAgent(action_num=env.action_num)
    heuristic_agent = NanoOFCPPerfectInfoAgent(action_num=env.action_num, use_raw=False, alpha=0.5)
    env.set_agents([agent, heuristic_agent])
    eval_env_random.set_agents([agent, random_agent])
    eval_env_heur.set_agents([agent, heuristic_agent])

    # Init a Logger to plot the learning curve, and use the name of the model so we can 
    # later plot these.
    logger_vs_random = Logger(log_dir, csv_name="dqn_vs_random.csv")
    logger_vs_heur = Logger(log_dir, csv_name="dqn_vs_heur.csv")

    best_score = 0

    for episode in range(episode_num):

        # Generate data from the environment
        trajectories, _ = env.heuristic_agent_run(is_training=True)

        # Feed transitions into agent memory, and train the agent
        for ts in trajectories[0]:
            agent.feed(ts)
        # for ts in trajectories[1]:
        #     agent.feed(ts)

        # Evaluate the performance. Play with random agents.
        if episode % evaluate_every == 0:
            
            tour_score_random = tournament(eval_env_random, evaluate_num)[0]
            if tour_score_random > best_score:
                state_dict = agent.get_state_dict()
                torch.save(state_dict, os.path.join(save_dir, 'best_model.pth'))
                best_score = tour_score_random
                logger_vs_random.log(str(env.timestep) + "  Saving best model. Expected Reward: " + str(best_score))

            tour_score_heur = heuristic_agent_tournament(eval_env_heur, evaluate_num)[0]

            logger_vs_random.log_performance(env.timestep, tour_score_random)
            logger_vs_heur.log_performance(env.timestep, tour_score_heur)

    # Close files in the logger
    logger_vs_random.close_files()
    logger_vs_heur.close_files()

    # Plot the learning curve
    logger_vs_random.plot('DQN')
    logger_vs_heur.plot('DQN')

    # Save model
    state_dict = agent.get_state_dict()
    torch.save(state_dict, os.path.join(save_dir, 'model.pth'))

    # Once model is saved, we can then test again to see how close the q values are to those 
    # which we sample from chosen games.
    q_value_log_dir = log_dir + 'q_values_logs/'
    mean_q_value_diffs = eval_q_value_approx(agent, random_agent, sample_size=20, num_rollouts=100, log_dir=q_value_log_dir)



if __name__ == '__main__':
    for i in range(1,2):
        training_run(log_dir = f".ow_model/experiments/nano_ofcp_dqn_with_heur/run{i}/logs/", 
        save_dir = f".ow_model/experiments/nano_ofcp_dqn_with_heur/run{i}/model/", random_seed=i*100)
