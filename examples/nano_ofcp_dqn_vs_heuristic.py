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

# Make environment
env = rlcard.make('nano_ofcp', config={'seed': 0})
eval_env_random = rlcard.make('nano_ofcp', config={'seed': 0})
eval_env_heur = rlcard.make('nano_ofcp', config={'seed': 0})

# Set the iterations numbers and how frequently we evaluate performance
evaluate_every = 1000
evaluate_num = 2500
episode_num = 20000

# The intial memory size
memory_init_size = 1000

# Train the agent every X steps
train_every = 1

# The paths for saving the logs and learning curves
# log_dir = '/content/drive/MyDrive/msc_thesis/nano_ofcp_models/one_hot_encoding_dqn_vs_heur_only_heur/logs/'
log_dir = './ow_models/nano_ofcp_dqn_vs_heur_only_heur/logs'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Set up the model saving folder.
save_dir = './ow_models/nano_ofcp_dqn_vs_heur_only_heur/models'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Set a global seed
set_global_seed(0)


# Set up the agents
agent_1 = DQNAgent(scope='dqn',
                action_num=env.action_num,
                replay_memory_init_size=memory_init_size,
                train_every=train_every,
                state_shape=env.state_shape,
                mlp_layers=[128, 128],
                device=torch.device('cpu'),
                epsilon_decay_steps=episode_num * 3,
                epsilon_start=0.9,
                epsilon_end=0.05,
                learning_rate=10e-4, 
                update_target_estimator_every=2000,
                discount_factor=1.0)

h_agent = NanoOFCPPerfectInfoAgent(action_num=env.action_num, use_raw=False, alpha=1)
r_agent = RandomAgent(action_num=env.action_num)
env.set_agents([agent_1, h_agent])
eval_env_random.set_agents([agent_1, r_agent])
eval_env_heur.set_agents([agent_1, h_agent])

# Initialize a Logger to plot the learning curve
logger = Logger(log_dir)

for episode in range(episode_num):

    # Generate data from the environment
    trajectories, _ = env.heuristic_agent_run(is_training=True)

    # print("Full Traj")
    # print(trajectories)

    # Feed transitions into agent memory, and train the agent
    # for ts in trajectories[0]:
    #     # print("feeding agent with trahj 0")
    #     # print(ts)
    #     print("Traj at index 0")
    #     print(ts)
    #     agent_1.feed(ts)

    # for i in range(env.player_num):
    for ts in trajectories[1]:
        agent_1.feed(ts)

    # Evaluate the performance. Play with random agents.
    if episode % evaluate_every == 0:
        print("Epsilon Value: {}".format(agent_1.epsilons[min(agent_1.total_t, agent_1.epsilon_decay_steps-1)]))
        logger.log_performance_using_env(env.timestep, "Heuristic", heuristic_agent_tournament(eval_env_heur, evaluate_num)[0])
        logger.log_performance_using_env(env.timestep, "Random", heuristic_agent_tournament(eval_env_random, evaluate_num)[0])


# Close files in the logger
logger.close_files()

# Plot the learning curve
logger.plot('nano_ofcp')

# Save model
state_dict = agent_1.get_state_dict()
torch.save(state_dict, os.path.join(save_dir, 'model.pth'))

