''' An example of learning a Deep-Q Agent on Nano OFCP using a PyTorch implimentation 
'''
import torch
import os

import rlcard
from rlcard.utils import set_global_seed, tournament
from rlcard.utils import Logger
from examples.nano_ofcp_q_value_approx import eval_q_value_approx
from rlcard.agents import DQNAgentPytorchNeg as DQNAgentNeg, NanoOFCPPerfectInfoAgent as HeurisitcAgent, RandomAgent
from nano_ofcp_heuristic_vs_random import heuristic_agent_tournament

def training_run(evaluate_every = 1000, 
                evaluate_num = 2500, 
                episode_num = 100000, 
                memory_init_size = 2500, 
                train_every = 1, 
                log_dir = None,
                save_dir = None,
                random_seed = 0):

    # Make environment
    env = rlcard.make('nano_ofcp', config={'seed': random_seed})
    env_heur_025 = rlcard.make('nano_ofcp', config={'seed': random_seed})
    env_heur_05 = rlcard.make('nano_ofcp', config={'seed': random_seed})
    env_heur_075 = rlcard.make('nano_ofcp', config={'seed': random_seed})
    env_heur_1 = rlcard.make('nano_ofcp', config={'seed': random_seed})
    eval_env_random = rlcard.make('nano_ofcp', config={'seed': random_seed})


    # The paths for saving the logs and learning curves
    if log_dir is None:
        log_dir = '.ow_model/experiments/nano_ofcp_dqn_result_exper/'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Set up the model saving folder.
    if save_dir is None:
        save_dir = '.ow_model/models/nano_ofcp_dqn_result_exper/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Set a global seed
    set_global_seed(random_seed)

    agent = DQNAgentNeg(scope='dqn',
                    action_num=env.action_num,
                    replay_memory_init_size=memory_init_size,
                    train_every=train_every,
                    state_shape=env.state_shape,
                    mlp_layers=[64, 64],
                    device=torch.device('cpu'),
                    epsilon_decay_steps=episode_num,
                    epsilon_start=0.6,
                    epsilon_end=0.1,
                    learning_rate=0.0001,
                    update_target_estimator_every=evaluate_every,
                    verbose=False, 
                    batch_size=32,
                    discount_factor=1.0,
                    max_neg_reward=-2)

    random_agent = RandomAgent(action_num=eval_env_random.action_num)
    heuristic_agent_025 = HeurisitcAgent(action_num=env.action_num, use_raw=False, alpha=0.25)
    heuristic_agent_05 = HeurisitcAgent(action_num=env.action_num, use_raw=False, alpha=0.5)
    heuristic_agent_075 = HeurisitcAgent(action_num=env.action_num, use_raw=False, alpha=0.75)
    heuristic_agent_1 = HeurisitcAgent(action_num=env.action_num, use_raw=False, alpha=1)

    env.set_agents([agent, random_agent])
    eval_env_random.set_agents([agent, random_agent])
    env_heur_025.set_agents([agent, heuristic_agent_025])
    env_heur_05.set_agents([agent, heuristic_agent_05])
    env_heur_075.set_agents([agent, heuristic_agent_075])
    env_heur_1.set_agents([agent, heuristic_agent_1])


    # Init a Logger to plot the learning curve, and use the name of the model so we can 
    # later plot these.
    logger = Logger(log_dir, csv_name="dqn_curr_learning.csv")

    # Display infomation about the agents networks.
    # print("Agents network shape: {}".format(agent.q_estimator.qnet))
    # print("Agents network layers: {}".format(agent.q_estimator.qnet.fc_layers))

    best_score = 0

    for episode in range(episode_num):

        if episode < 40000:
            # Generate data from the environment
            trajectories, _ = env.run(is_training=True)
        elif episode < 60000:
            trajectories, _ = env_heur_025.heuristic_agent_run(is_training=True)
        elif episode < 75000:
            trajectories, _ = env_heur_05.heuristic_agent_run(is_training=True)
        elif episode < 90000:
            trajectories, _ = env_heur_075.heuristic_agent_run(is_training=True)
        else:
            trajectories, _ = env_heur_1.heuristic_agent_run(is_training=True)


        # Feed transitions into agent memory, and train the agent
        for ts in trajectories[0]:
            agent.feed(ts)

        # Evaluate the performance. Play with random agents.
        if episode % evaluate_every == 0:
            
            tour_score = tournament(eval_env_random, evaluate_num)[0]
            
            if tour_score > best_score:
                state_dict = agent.get_state_dict()
                torch.save(state_dict, os.path.join(save_dir, 'best_model.pth'))
                best_score = tour_score
                logger.log(str(episode) + "  Saving best model. Expected Reward: " + str(best_score))

            logger.log_performance(episode, tour_score)

    # Close files in the logger
    logger.close_files()

    # Plot the learning curve
    logger.plot('dqn_curr')

    # Save model
    state_dict = agent.get_state_dict()
    torch.save(state_dict, os.path.join(save_dir, 'model.pth'))

    # Once model is saved, we can then test again to see how close the q values are to those 
    # which we sample from chosen games.
    q_value_log_dir = log_dir + 'q_values_logs/'
    mean_q_value_diffs = eval_q_value_approx(agent, random_agent, sample_size=20, num_rollouts=100, log_dir=q_value_log_dir)



if __name__ == '__main__':
    for i in range(1,2):
        training_run(log_dir = f".ow_model/experiments/nano_ofcp_dqn_curr_learning/run{i}/logs/", 
        save_dir = f".ow_model/experiments/nano_ofcp_dqn_curr_learning/run{i}/model/", random_seed=i*100)
