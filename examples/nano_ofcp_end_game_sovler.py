''' An example of learning a Deep-Q Agent on Nano OFCP using a PyTorch implimentation 
'''
import torch
import os
import rlcard
import json
import pickle 

from rlcard.agents import DQNAgentEndGame
from rlcard.agents import RandomAgent
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
    env = rlcard.make('nano_ofcp', config={'seed': random_seed, 'allow_step_back': True})
    eval_env = rlcard.make('nano_ofcp', config={'seed': random_seed, 'allow_step_back': True})

    # The paths for saving the logs and learning curves
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Set up the model saving folder.
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Save the arguments for the q_agent so these can be loaded easily.
    json_file = open(os.path.join(save_dir, 'agent_kwargs.json'), "w")
    json.dump(q_agent_kwargs, json_file)
    json_file.close()

    # Set a global seed
    set_global_seed(random_seed)

    dqn_end_game_agent = DQNAgentEndGame(**q_agent_kwargs)
    random_agent = RandomAgent(action_num=eval_env.action_num)

    # Fill up the end game solver if this is already been written to.
    if os.path.exists(os.path.join(save_dir, 'end_game_solver.pkl')):
        print("Loading from pickle file.")
        pickle_file = open(os.path.join(save_dir, 'end_game_solver.pkl'), "rb")
        dqn_end_game_agent.end_game_solver = pickle.load(pickle_file)

    env.set_agents([dqn_end_game_agent, random_agent])
    eval_env.set_agents([dqn_end_game_agent, random_agent])

    # Init a Logger to plot the learning curve, and use the name of the model so we can 
    # later plot these.
    logger = Logger(log_dir, csv_name="dqn.csv")

    best_score = 0

    for episode in range(episode_num):

        if episode % 25000:
            # Save the end game solver dictionary
            pickle_file = open(os.path.join(save_dir, 'end_game_solver.pkl'), "wb")
            pickle.dump(dqn_end_game_agent.end_game_solver, pickle_file)
            pickle_file.close()


        # Generate data from the environment
        trajectories, _ = env.end_game_agent_run(is_training=True)

        # Feed transitions into agent memory, and train the agent
        for ts in trajectories[0]:
            dqn_end_game_agent.feed(ts)

        # Evaluate the performance. Play with random agents.
        if episode % evaluate_every == 0:
            
            tour_score = tournament(eval_env, evaluate_num)[0]
            if tour_score > best_score:
                state_dict = dqn_end_game_agent.get_state_dict()
                torch.save(state_dict, os.path.join(save_dir, 'best_model.pth'))
                best_score = tour_score
                logger.log("\n" +str(env.timestep) + "  Saving best model. Expected Reward: " + str(best_score))

            logger.log_performance(env.timestep, tour_score)

    # Close files in the logger
    logger.close_files()

    # Plot the learning curve
    logger.plot('DQN')

    # Save model
    state_dict = dqn_end_game_agent.get_state_dict()
    torch.save(state_dict, os.path.join(save_dir, 'model.pth'))

    # We want to save the dictionary we've built up so far as to not need to Recompute
    # the same values. So we can do this by chucking it into a JSON file.
    # Save the arguments for the q_agent so these can be loaded easily.
    
    # At the moment this overwrites anything we had, we can change to append but will 
    # duplicate what was already there.
    # end_game_file = open(os.path.join(save_dir, 'end_game_solver.json'), "w")
    # json.dump(dqn_end_game_agent.end_game_solver, end_game_file)
    # end_game_file.close()


    pickle_file = open(os.path.join(save_dir, 'end_game_solver.pkl'), "wb")
    pickle.dump(dqn_end_game_agent.end_game_solver, pickle_file)
    pickle_file.close()

    # Once model is saved, we can then test again to see how close the q values are to those 
    # which we sample from chosen games.
    # q_value_log_dir = log_dir + 'q_values_logs/'
    # mean_q_value_diffs = eval_q_value_approx(agent, random_agent, sample_size=20, num_rollouts=100, log_dir=q_value_log_dir)


if __name__ == '__main__':

    for i in range(0,1):

        run_kwargs = {
            'evaluate_every': 5000, 
            'evaluate_num': 5000, 
            'episode_num': 100000, 
            'random_seed': 1
        }

        agent_kwargs = {
            'scope': 'dqn',
            'state_shape': 108,
            'action_num': 12,
            'device': 'cpu',
            'replay_memory_init_size': 1000,
            'update_target_estimator_every': 2500,
            'train_every': 2,
            'mlp_layers': [128, 128],
            'learning_rate': 0.00005,
            'batch_size': 64,
            'epsilon_start': 0.5,
            'epsilon_end': 0.05,
            'epsilon_decay_steps': run_kwargs['episode_num'],
            'discount_factor': 1.0,
            'verbose': False,
            'max_neg_reward': -2, 
        }

        training_run(
            log_dir=f"ow_model/experiments/nano_ofcp_end_game/run{i}/logs/", 
            save_dir=f"ow_model/experiments/nano_ofcp_end_game/run{i}/model/",
            q_agent_kwargs=agent_kwargs,
            **run_kwargs
        )