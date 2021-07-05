''' An example of loading pre-trained NFSP model on Leduc Holdem
'''
import os
import torch
import json
import rlcard
from rlcard.agents import DQNAgentPytorch as DQNAgent, DQNAgentPytorchNeg as DQNAgentNeg
from rlcard.agents.random_agent import RandomAgent
from rlcard.utils.utils import set_global_seed, tournament
from rlcard.utils import Logger
from examples.nano_ofcp_q_value_approx import eval_q_value_approx

import matplotlib.pyplot as plt


def play_tournament(env, evaluate_num):
    reward = tournament(env, evaluate_num)[0]
    print('Average reward against random agent: ', reward)

def env_load_dqn_agent_and_random_agent(agent_kwargs, agent_type, agent_path):

    # Make environment
    env = rlcard.make('nano_ofcp', config={'seed': 0})

    # Create an object which will then fill with the correct elements.
    if agent_type == "DQN":
        agent = DQNAgent(**agent_kwargs)
    elif agent_type == "DQN_NEG":
        agent = DQNAgentNeg(**agent_kwargs)
    else:
        raise Exception("Invalid type of class for the DQN loading algorithm.")

    random_agent = RandomAgent(action_num=env.action_num)
    
    # Load in the weights and the opt to continue training or evaluate as required.
    checkpoint = torch.load(agent_path)
    agent.load(checkpoint)

    # Set the environment to include the agents and return this.
    env.set_agents([agent, random_agent])
    return env

def load_dqn_agent(agent_kwargs, agent_type, agent_path):

    # Create an object which will then fill with the correct elements.
    if agent_type == "DQN":
        agent = DQNAgent(**agent_kwargs)
    elif agent_type == "DQN_NEG":
        agent = DQNAgentNeg(**agent_kwargs)
    else:
        raise Exception("Invalid type of class for the DQN loading algorithm.")

    # Load in the weights and the opt to continue training or evaluate as required.
    checkpoint = torch.load(agent_path)
    agent.load(checkpoint)

    return agent


def training_run(env, log_dir, save_dir, evaluate_every, evaluate_num, episode_num, random_seed):

    # Create the normal and eval environment, and populate this with the agent from ours and a 
    # random agent.
    random_agent = RandomAgent(action_num=env.action_num)
    eval_env = rlcard.make('nano_ofcp', config={'seed': random_seed})
    eval_env.set_agents([env.agents[0], random_agent])

    # Set the random seed.
    set_global_seed(random_seed)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Set up the model saving folder.
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Init a Logger to plot the learning curve, and use the name of the model so we can 
    # later plot these.
    logger = Logger(log_dir, csv_name="dqn_cycle_2.csv")
    best_score = 0

    for episode in range(episode_num):

        # Generate data from the environment
        trajectories, _ = env.run(is_training=True)

        # Feed transitions into agent memory, and train the agent
        for ts in trajectories[0]:
            env.agents[0].feed(ts)

        # Evaluate the performance. Play with random agents.
        if episode % evaluate_every == 0:
            
            tour_score = tournament(eval_env, evaluate_num)[0]
            if tour_score > best_score:
                state_dict = env.agents[0].get_state_dict()
                torch.save(state_dict, os.path.join(save_dir, 'best_model.pth'))
                best_score = tour_score
                logger.log(str(env.timestep) + "  Saving best model. Expected Reward: " + str(best_score))

            logger.log_performance(env.timestep, tournament(eval_env, evaluate_num)[0])

    # Close files in the logger
    logger.close_files()

    # Plot the learning curve
    logger.plot('DQN')

    # Save model
    state_dict = env.agents[0].get_state_dict()
    torch.save(state_dict, os.path.join(save_dir, 'model.pth'))

    # Once model is saved, we can then test again to see how close the q values are to those 
    # which we sample from chosen games.
    q_value_log_dir = log_dir + 'q_values_logs/'
    mean_q_value_diffs = eval_q_value_approx(env.agents[0], random_agent, sample_size=20, num_rollouts=100, log_dir=q_value_log_dir)



# Agent types which are allowed are DQN, DQN_NEG, DQN_UCB.

def load_env(directory, trainable, agent_type): 

    agent_path = directory +  "model/model.pth"
    json_kwargs_path = directory + "model/agent_kwargs.json"

    json_file = open(json_kwargs_path, "r")
    agent_kwargs = json.load(json_file)

    if trainable:
        greedy_parameters = {
            'epsilon_start': 0.1,
            'epsilon_end': 0.1,
            'epsilon_decay_steps': 1,
        }
    else:
        greedy_parameters = {
            'epsilon_start': 0.0,
            'epsilon_end': 0.0,
            'epsilon_decay_steps': 1,
        }

    # We can overwrite any parameters we require using the update to a dictionary method.
    agent_kwargs.update(greedy_parameters)

    return env_load_dqn_agent_and_random_agent(
        agent_kwargs,
        agent_type,
        agent_path
    )


def continue_training(directory, agent_type):

    # We need to form the log and save directory
    log_dir = directory + "logs_cycle_2"
    save_dir = directory + "model_cycle_2"

    run_kwargs = {
        'evaluate_every': 500, 
        'evaluate_num': 1000, 
        'episode_num': 2000, 
        'random_seed': 0
    }

    env = load_env(directory, trainable=True, agent_type=agent_type)
    training_run(env, log_dir, save_dir, **run_kwargs)


if __name__ == '__main__':
    directory = "ow_model/experiments/nano_ofcp_dqn_vs_random_training_saving/run0/"
    agent_type = "DQN"
    continue_training(directory, agent_type)