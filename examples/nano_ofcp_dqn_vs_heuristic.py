''' 
An example of learning a Deep-Q Agent on Nano_OFCP.
'''

import tensorflow as tf
import os

import rlcard
from rlcard.agents import DQNAgent, NanoOFCPPerfectInfoAgent, RandomAgent
from rlcard.utils import set_global_seed, tournament
from rlcard.utils import Logger
from nano_ofcp_heuristic_vs_random import heuristic_agent_tournament

# Make environment
env = rlcard.make('nano_ofcp', config={'seed': 0})
eval_env = rlcard.make('nano_ofcp', config={'seed': 0})

# Set the iterations numbers and how frequently we evaluate performance
evaluate_every = 500
evaluate_num = 250
episode_num = 5000

# The intial memory size
memory_init_size = 100

# Train the agent every X steps
train_every = 1

# The paths for saving the logs and learning curves
log_dir = './ow_experiments/nano_ofcp_dqn_vs_heur_result/'

# Set a global seed
set_global_seed(0)

with tf.Session() as sess:

    # Initialize a global step
    global_step = tf.Variable(0, name='global_step', trainable=False)

    print(env.state_shape)

    # Set up the agents
    agent_1 = DQNAgent(sess,
                     scope='dqn_1',
                     action_num=env.action_num,
                     replay_memory_init_size=memory_init_size,
                     train_every=train_every,
                     state_shape=env.state_shape,
                     mlp_layers=[128, 128])
    h_agent = NanoOFCPPerfectInfoAgent(action_num=eval_env.action_num, use_raw=True, alpha=0.25)
    r_agent = RandomAgent(action_num=eval_env.action_num)
    env.set_agents([agent_1, h_agent])
    eval_env.set_agents([agent_1, r_agent])

    # Initialize global variables
    sess.run(tf.global_variables_initializer())

    # Initialize a Logger to plot the learning curve
    logger = Logger(log_dir)

    for episode in range(episode_num):

        # Generate data from the environment
        trajectories, _ = env.heuristic_agent_run(is_training=True)

        # Feed transitions into agent memory, and train the agent
        for ts in trajectories[0]:
            agent_1.feed(ts)

        # Evaluate the performance. Play with random agents.
        if episode % evaluate_every == 0:
            logger.log_performance(env.timestep, heuristic_agent_tournament(eval_env, evaluate_num)[0])

    # Close files in the logger
    logger.close_files()

    # Plot the learning curve
    logger.plot('nano_ofcp')
    
    # Save model
    save_dir = './ow_models/nano_ofcp_dqn_vs_heur'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    saver = tf.train.Saver()
    saver.save(sess, os.path.join(save_dir, 'model'))
    
