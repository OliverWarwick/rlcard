''' An example of learning a Deep-Q Agent on Nano OFCP using a PyTorch implimentation 
'''
import torch
import os

import rlcard
from rlcard.agents import DQNAgentPytorch as DQNAgent
from rlcard.agents import RandomAgent
from rlcard.utils import set_global_seed, tournament
from rlcard.utils import Logger

def training_run(evaluate_every = 1000, 
                evaluate_num = 2500, 
                episode_num = 20000, 
                memory_init_size = 1000, 
                train_every = 1, 
                log_dir = None,
                save_dir = None):

    # Make environment
    env = rlcard.make('nano_ofcp', config={'seed': 0})
    eval_env = rlcard.make('nano_ofcp', config={'seed': 0})

    # The paths for saving the logs and learning curves
    if log_dir is None:
        log_dir = './experiments/nano_ofcp_dqn_result/'
    if not os.path.exists(log_dir):
            os.makedirs(log_dir)

    # Set up the model saving folder.
    if save_dir is None:
        save_dir = 'models/nano_dqn_pytorch'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Set a global seed
    set_global_seed(0)

    agent = DQNAgent(scope='dqn',
                    action_num=env.action_num,
                    replay_memory_init_size=memory_init_size,
                    train_every=train_every,
                    state_shape=env.state_shape,
                    mlp_layers=[64, 64],
                    device=torch.device('cpu'),
                    epsilon_decay_steps=50000)
    random_agent = RandomAgent(action_num=eval_env.action_num)
    env.set_agents([agent, random_agent])
    eval_env.set_agents([agent, random_agent])

    # Init a Logger to plot the learning curve
    logger = Logger(log_dir)

    # Display infomation about the agents networks.
    # print("Agents network shape: {}".format(agent.q_estimator.qnet))
    # print("Agents network layers: {}".format(agent.q_estimator.qnet.fc_layers))

    best_score = 0

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
                logger.log(str(env.timestep) + "  Saving best model. Accuracy: " + str(best_score))

            logger.log_performance(env.timestep, tournament(eval_env, evaluate_num)[0])

    # Close files in the logger
    logger.close_files()

    # Plot the learning curve
    logger.plot('DQN')

    # Save model

    state_dict = agent.get_state_dict()
    print(state_dict.keys())
    torch.save(state_dict, os.path.join(save_dir, 'model.pth'))

if __name__ == '__main__':
    training_run()
