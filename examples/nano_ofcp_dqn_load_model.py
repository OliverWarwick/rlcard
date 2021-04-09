''' An example of loading a pre-trained NFSP model on Leduc Hold'em
'''
import tensorflow as tf
import os

import rlcard
from rlcard.agents import DQNAgent
from rlcard.agents import RandomAgent
from rlcard.utils import set_global_seed, tournament

# Make environment
env = rlcard.make('nano_ofcp', config={'seed': 0})

# Set a global seed
set_global_seed(0)

# Load pretrained model
graph = tf.Graph()
sess = tf.Session(graph=graph)

with graph.as_default():
    agent = DQNAgent(sess,
            scope='dqn_1',
            action_num=env.action_num,
            state_shape=env.state_shape,
            mlp_layers=[64, 64])
    random_agent = RandomAgent(action_num=env.action_num)

# We have a pretrained model here. Change the path for your model.
check_point_path = os.path.join(rlcard.__path__[0], 'models/pretrained/leduc_holdem_nfsp')

with sess.as_default():
    with graph.as_default():
        saver = tf.train.Saver()
        saver.restore(sess, tf.train.latest_checkpoint(check_point_path))

# Evaluate the performance. Play with random agents.
evaluate_num = 1000
random_agent = RandomAgent(env.action_num)
env.set_agents([agent, random_agent])
reward = tournament(env, evaluate_num)[0]
print('Average reward against random agent: ', reward)

