''' An example of learning a Deep-Q Agent on Nano OFCP using a PyTorch implimentation 
'''
import torch
import os
import numpy as np
import pandas as pd
import sys
import json

import rlcard
from rlcard.agents import DQNAgentPytorchUCB as DQNAgentUCB, RandomAgent
from rlcard.utils import set_global_seed, tournament
from rlcard.utils import Logger
from examples.nano_ofcp_q_value_approx import eval_q_value_approx


def training_run(log_dir, 
    save_dir, 
    q_agent_kwargs, 
    evaluate_every,
    evaluate_num,
    episode_num,
    random_seed):

    # Make environment
    env = rlcard.make('nano_ofcp', config={'seed': random_seed})
    eval_env = rlcard.make('nano_ofcp', config={'seed': random_seed})

    # The paths for saving the logs and learning curves
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Set up the model saving folder.
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Set a global seed
    set_global_seed(random_seed)

    agent = DQNAgentUCB(**q_agent_kwargs)
    random_agent = RandomAgent(action_num=eval_env.action_num)

    env.set_agents([agent, random_agent])
    eval_env.set_agents([agent, random_agent])

    # Init a Logger to plot the learning curve, and use the name of the model so we can 
    # later plot these.
    logger = Logger(log_dir, csv_name="dqn_ucb_handcrafted.csv")

    best_score = 0
    num_states = []

    for episode in range(episode_num):

        # Generate data from the environment
        trajectories, _ = env.run(is_training=True)

        # Feed transitions into agent memory, and train the agent
        for ts in trajectories[0]:
            agent.feed(ts)

        # Evaluate the performance. Play with random agents.
        if episode % evaluate_every == 0:
            
            tour_score = tournament(eval_env, evaluate_num)[0]
            if tour_score > best_score:
                state_dict = agent.get_state_dict()
                torch.save(state_dict, os.path.join(save_dir, 'best_model.pth'))
                best_score = tour_score
                logger.log(str(env.timestep) + "  Saving best model. Expected Reward: " + str(best_score))

            logger.log_performance(env.timestep, tournament(eval_env, evaluate_num)[0])

        if episode in [0, 10, 100, 500, 1000, 2500, 5000, 10000, 25000]:
            print("\nSummary Stats on Map:")
            print("Size in bytes: {}".format(sys.getsizeof(agent.count_map)))
            print("MByes: {}".format(sys.getsizeof(agent.count_map) / 1024))
            print("Length of hashtable: {}".format(len(agent.count_map.keys())))

            if episode != 0:
                new_state_value = (len(agent.count_map) - num_states[-1][1]) / ((episode - num_states[-1][0]) * 12)
                num_states.append([episode, len(agent.count_map), new_state_value])
            else:
                num_states.append([episode, len(agent.count_map), 0])

    # Close files in the logger
    logger.close_files()

    # Plot the learning curve
    logger.plot('DQN_UCB_HANDCRAFTED')

    # Save model
    state_dict = agent.get_state_dict()
    torch.save(state_dict, os.path.join(save_dir, 'model.pth'))

    # Save the arguments for the q_agent so these can be loaded easily.
    json_file = open(os.path.join(save_dir, 'agent_kwargs.json'), "w")
    json.dump(q_agent_kwargs, json_file)
    json_file.close()

    print("Number of keys required at each stage.")
    print(num_states)

    return num_states


if __name__ == '__main__':

    overall_df = pd.DataFrame([], columns=['episode', 'states', 'prop'])
    i = 0
    for i in range(0,1):

        run_kwargs = {
            'evaluate_every': 2500, 
            'evaluate_num': 5000, 
            'episode_num': 10000, 
            'random_seed': i
        }

        agent_kwargs = {
            'scope': 'dqn',
            'state_shape': 108,
            'action_num': 12,
            'device': 'cpu',
            'replay_memory_init_size': 1000,
            'update_target_estimator_every': 2500,
            'train_every': 1,
            'mlp_layers': [128, 128],
            'learning_rate': 0.00005,
            'batch_size': 64,
            'epsilon_start': 1.0,
            'epsilon_end': 0.1,
            'epsilon_decay_steps': run_kwargs['episode_num'],
            'discount_factor': 1.0,
            'verbose': False,
            'optimisitic_bias_on_action': 4,
            'optimisitic_bias_on_bootstrap': 4,
            'optimism_decay': 0.5
        }

        l = training_run(
            log_dir=f"ow_model/experiments/nano_ofcp_dqn_ucb_exp_long_run/run{i}/logs/",
            save_dir=f"ow_model/experiments/nano_ofcp_dqn_ucb_exp_long_run/run{i}/model/",
            q_agent_kwargs=agent_kwargs,
            **run_kwargs
        )

        df = pd.DataFrame(l, columns=['episode', 'states', 'prop'])

        overall_df = overall_df.append(df, ignore_index=True)

    print("Dataframe of explored states.")    
    print(overall_df)
    overall_df.to_csv('state_data_small_key_exp_run.csv')


       