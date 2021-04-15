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
eval_env = rlcard.make('nano_ofcp', config={'seed': 0})

# Set the iterations numbers and how frequently we evaluate/save plot
evaluate_every = 5000
evaluate_num = 5000
episode_num = 20000

# The intial memory size
memory_init_size = 1000

# Train the agent every X steps
train_every = 64

# The paths for saving the logs and learning curves
log_dir = './experiments/nano_ofcp_nfsp_result/'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Set up the model saving folder.
save_dir = './models/nano_ofcp_nfsp_result/'
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
                      hidden_layers_sizes=[128,128],
                      min_buffer_size_to_learn=memory_init_size,
                      q_replay_memory_init_size=memory_init_size,
                      train_every=train_every,
                      q_train_every=train_every,
                      q_mlp_layers=[128,128],
                    device=torch.device('cpu'))
    agents.append(agent)
random_agent = RandomAgent(action_num=eval_env.action_num)

env.set_agents(agents)
eval_env.set_agents([agents[0], random_agent])

# Init a Logger to plot the learning curve
logger = Logger(log_dir)

best_score = 0

for episode in range(episode_num):

    # First sample a policy for the episode
    for agent in agents:
        agent.sample_episode_policy()

    # Generate data from the environment
    trajectories, _ = env.run(is_training=True)
    if episode % evaluate_every == 0:
        logger.log("Traj:")
        logger.log(str(trajectories))

    # Feed transitions into agent memory, and train the agent
    for i in range(env.player_num):
        for ts in trajectories[i]:
            agents[i].feed(ts)

    # Evaluate the performance. Play with random agents.
    if episode % evaluate_every == 0:
    
        tour_score = tournament(eval_env, evaluate_num)[0]
        if tour_score > best_score:

            state_dict = agent.get_state_dict()

            torch.save(state_dict, os.path.join(save_dir, 'best_model.pth'))

            best_score = tour_score
            logger.log(str(env.timestep) + "  Saving best model. Accuracy: " + str(best_score))

        logger.log_performance(env.timestep, tour_score)


# Close files in the logger
logger.close_files()

# Plot the learning curve
logger.plot('NFSP')

# Save model
state_dict = {}
for agent in agents:
    state_dict.update(agent.get_state_dict())
torch.save(state_dict, os.path.join(save_dir, 'model.pth'))
