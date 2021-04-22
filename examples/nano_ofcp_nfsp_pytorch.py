''' An example of learning a NFSP Agent on Leduc Holdem
'''
import os
import torch

import rlcard
from rlcard.agents import NFSPAgentPytorch as NFSPAgent
from rlcard.agents import RandomAgent
from rlcard.utils import set_global_seed, tournament
from rlcard.utils import Logger

# Make environment
env = rlcard.make('nano_ofcp', config={'seed': 0})
eval_env_0 = rlcard.make('nano_ofcp', config={'seed': 0})
eval_env_1 = rlcard.make('nano_ofcp', config={'seed': 0})

# Set the iterations numbers and how frequently we evaluate/save plot
evaluate_every = 2000
evaluate_num = 5000
episode_num = 2500
# The intial memory size
memory_init_size = 1000

# Train the agent every X steps
train_every = 16

# The paths for saving the logs and learning curves
# log_dir = '/content/drive/MyDrive/msc_thesis/nano_ofcp_models/nfsp_low_lr/logs'
log_dir = './ow_models/base'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Set up the model saving folder.
# save_dir = '/content/drive/MyDrive/msc_thesis/nano_ofcp_models/nfsp_low_lr/models'
save_dir = './ow_models/base'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Set a global seed
set_global_seed(0)

# Set agents
agents = []
for i in range(env.player_num):
    agent = NFSPAgent(scope='nfsp' + str(i),
                      action_num=env.action_num,
                      state_shape=env.state_shape,
                      hidden_layers_sizes=[64, 64],
                      min_buffer_size_to_learn=memory_init_size,
                      q_replay_memory_init_size=memory_init_size,
                      train_every=train_every,
                      q_train_every=train_every,
                      q_mlp_layers=[64, 64],
                      q_discount_factor=1.0,
                      q_epsilon_start=0.25,
                      device=torch.device('cpu'),
                      rl_learning_rate=0.005,
                      evaluate_with='best_response')
    agents.append(agent)
random_agent = RandomAgent(action_num=eval_env_0.action_num)

env.set_agents(agents)
eval_env_0.set_agents([agents[0], random_agent])
eval_env_1.set_agents([agents[1], random_agent])

# Init a Logger to plot the learning curve
logger = Logger(log_dir)

best_score = 0

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

        if tour_score_player_0_br > best_score:

            state_dict = {}
            for agent in agents:
                state_dict.update(agent.get_state_dict())
            torch.save(state_dict, os.path.join(save_dir, 'best_model.pth'))
            best_score = tour_score_player_0_br
            logger.log(str(env.timestep) + "  Saving best model. Score vs Random Agent: " + str(best_score))

        logger.log_performance_using_algo("P0: Best Response", env.timestep, tour_score_player_0_br)
        logger.log_performance_using_algo("P0: Average Policy", env.timestep, tour_score_player_0_ap)
        logger.log_performance_using_algo("P1: Best Response", env.timestep, tour_score_player_1_br)
        logger.log_performance_using_algo("P1: Average Policy", env.timestep, tour_score_player_1_ap)

# Close files in the logger
logger.close_files()

# Plot the learning curve
logger.quad_plot('NFSP_BR_P0', 'NFSP_AVG_P0', 'NFSP_BR_P1', 'NFSP_AVG_P1')

# Save model
state_dict = {}
for agent in agents:
    state_dict.update(agent.get_state_dict())
torch.save(state_dict, os.path.join(save_dir, 'model.pth'))
