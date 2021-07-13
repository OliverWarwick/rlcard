''' An example of learning a Deep-Q Agent on Nano OFCP using a PyTorch implimentation 
'''
import torch
import os
import json 

import rlcard
from rlcard.utils import set_global_seed, tournament
from rlcard.utils import Logger
from examples.nano_ofcp_q_value_approx import eval_q_value_approx
from rlcard.agents import DQNAgentPytorchNeg as DQNAgentNeg, NanoOFCPPerfectInfoAgent as HeurisitcAgent, RandomAgent
from nano_ofcp_heuristic_vs_random import heuristic_agent_tournament
from nano_ofcp_dqn_pytorch_load_model import load_dqn_agent

def training_run(agent_kwargs, agent_type, agent_path, log_dir, save_dir, evaluate_every, evaluate_num, episode_num, random_seed):

    # Make environment
    # env = rlcard.make('nano_ofcp', config={'seed': random_seed})
    # env_heur_025 = rlcard.make('nano_ofcp', config={'seed': random_seed})
    env_heur_05 = rlcard.make('nano_ofcp', config={'seed': random_seed})
    # env_heur_075 = rlcard.make('nano_ofcp', config={'seed': random_seed})
    # env_heur_1 = rlcard.make('nano_ofcp', config={'seed': random_seed})
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

    agent = load_dqn_agent(agent_kwargs, agent_type, agent_path)

    random_agent = RandomAgent(action_num=eval_env_random.action_num)
    # heuristic_agent_025 = HeurisitcAgent(action_num=env.action_num, use_raw=False, alpha=0.25)
    heuristic_agent_05 = HeurisitcAgent(action_num=env_heur_05.action_num, use_raw=False, alpha=0.5)
    # heuristic_agent_075 = HeurisitcAgent(action_num=env.action_num, use_raw=False, alpha=0.75)
    # heuristic_agent_1 = HeurisitcAgent(action_num=env.action_num, use_raw=False, alpha=1)

    # env.set_agents([agent, random_agent])
    eval_env_random.set_agents([agent, random_agent])
    # env_heur_025.set_agents([agent, heuristic_agent_025])
    env_heur_05.set_agents([agent, heuristic_agent_05])
    # env_heur_075.set_agents([agent, heuristic_agent_075])
    # env_heur_1.set_agents([agent, heuristic_agent_1])


    # Init a Logger to plot the learning curve, and use the name of the model so we can 
    # later plot these.
    logger = Logger(log_dir, csv_name="dqn_curr_learning.csv")

    # Display infomation about the agents networks.
    # print("Agents network shape: {}".format(agent.q_estimator.qnet))
    # print("Agents network layers: {}".format(agent.q_estimator.qnet.fc_layers))

    best_score = 0

    for episode in range(episode_num):

        # if episode < 40000:
        #     # Generate data from the environment
        #     trajectories, _ = env.run(is_training=True)
        # elif episode < 60000:
        #     trajectories, _ = env_heur_025.heuristic_agent_run(is_training=True)
        # elif episode < 75000:
        #     trajectories, _ = env_heur_05.heuristic_agent_run(is_training=True)
        # elif episode < 90000:
        #     trajectories, _ = env_heur_075.heuristic_agent_run(is_training=True)
        # else:
        #     trajectories, _ = env_heur_1.heuristic_agent_run(is_training=True)

        trajectories, _ = env_heur_05.heuristic_agent_run(is_training=True)

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
    mean_q_value_diffs = eval_q_value_approx(agent, random_agent, sample_size=100, num_rollouts=100, log_dir=q_value_log_dir)



if __name__ == '__main__':

    trainable = True
    directory = "ow_model/experiments/nano_ofcp_dqn_neg_reg/run0/"

    # We need to form the log and save directory
    log_dir = directory + "logs_cycle_curr"
    save_dir = directory + "model_cycle_curr"

    run_kwargs = {
        'evaluate_every': 5000, 
        'evaluate_num': 5000, 
        'episode_num': 100000, 
        'random_seed': 0
    }
    
    agent_path = directory + "model/model.pth"
    agent_type = "DQN_NEG"

    json_kwargs_path = directory + "model/agent_kwargs.json"
    json_file = open(json_kwargs_path, "r")
    agent_kwargs = json.load(json_file)

    greedy_parameters = {
        'epsilon_start': 0.25,
        'epsilon_end': 0.1,
        'epsilon_decay_steps': run_kwargs['episode_num'],
    }

    agent_kwargs.update(greedy_parameters)
    
    training_run(agent_kwargs, agent_type, agent_path, log_dir, save_dir, **run_kwargs)
