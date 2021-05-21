import numpy as np
import random 
import math 
import time
from rlcard.games.nano_ofcp.ofcp_utils import STRING_TO_RANK, Row
from rlcard.utils.utils import init_mini_deck
from copy import deepcopy
import multiprocessing as mp
''' TODO: 
The function np_random will automatically roll forward to the random next value after each call so we do not need to worry about this.
'''


# TODO: MULTI THREAD THIS. 

class MCAgent:

    def __init__(self, action_num, use_raw):
        self.use_raw = use_raw
        self.action_num = action_num
        self.num_roll_outs = 5
        self.verbose = False


    
   
    def step(self, state, env, player_id): 
        ''' Multi threaded version ''' 
        
        start_time = time.time()
        num_roll_outs = env.game.dealer.deal_counter * 3

        # Failing all else, pick the first action possible.
        best_avg_reward, best_action = -1.0, state['legal_actions'][0]

        manager = mp.Manager()
        return_dict = manager.dict()
        jobs = []
        for action in state['legal_actions']:
            p = mp.Process(target=roll_out_for_action, args=(env, player_id, action, num_roll_outs, return_dict))
            jobs.append(p)
            p.start()
        for proc in jobs:
            proc.join()
        print(return_dict)
        print(max(return_dict, key=return_dict.get))
        print("Time taken: {}".format(time.time() - start_time))
        return max(return_dict, key=return_dict.get)



    
    # def step(self, state, env, player_id): 
    #     print(env.game.dealer.deal_counter)
    #     start_time = time.time()
    #     num_roll_outs = env.game.dealer.deal_counter * 2

    #     # Failing all else, pick the first action possible.
    #     best_avg_reward, best_action = -1.0, state['legal_actions'][0]
    #     if self.verbose: print("Call to MC Agent Step Function. \nCurrent state: {}".format(state))

    #     for action in state['legal_actions']:

    #         if self.verbose: print("Starting to try action: {}".format(action))

    #         avg_reward = 0.0
            
    #         # Step forwards
    #         next_state, next_player_id = env.step(action)
    #         # start_state = deepcopy(next_state)
    #         if self.verbose: print("Post action state: {}".format(next_state))
             
    #         for iteration in range(num_roll_outs):
    #             depth = 0

    #             while not env.is_over():
    #                 if self.verbose: print("Player ID: {}".format(next_player_id))
    #                 # Pick a random move using the current env, which will take a random choice of the legal actions, and then look up the index of this, because the get_legal_action function returns the human readable representation of these.
    #                 legal_acts = [env.actions.index(act) for act in env.game.get_legal_actions()]
    #                 if self.verbose: print("Possible legal actions: {}".format(legal_acts))
    #                 random_action = env.actions.index(random.choice(env.game.get_legal_actions()))
    #                 if self.verbose: print("Random action selected: {}".format(random_action))
    #                 next_state, next_player_id = env.step(random_action)
    #                 depth += 1
                
    #             if self.verbose: print("Final state: {}".format(next_state))
    #             payoff = env.get_payoffs()[player_id]
    #             if self.verbose: print("Payoff: {}".format(payoff))
    #             avg_reward = avg_reward + (payoff - avg_reward) / (iteration + 1)
    #             if self.verbose: print("new avg: {}".format(avg_reward))
    #             if self.verbose: print("Action: {}, Iter: {}, End Payoff: {}".format(action, iteration, payoff))

    #             if self.verbose: print("Steping back for {} iterations".format(depth))
    #             for _ in range(depth):
    #                 env.step_back()
    #             if self.verbose: print("Should be back to post action state: {}".format(env._extract_state(env.game.get_state(player_id))))

    #         if self.verbose:print("Avg reward for action {} is {}".format(action, avg_reward))
    #         if avg_reward > best_avg_reward:
    #             best_avg_reward, best_action = avg_reward, action 

    #         # Final set back for next action.
    #         env.step_back()
    #         if self.verbose: print("Inital State: {}\n Starting over state: {}".format(state, env._extract_state(env.game.get_state(player_id))))
    #     # print()
    #     print("Time taken to execute one step call: {}".format(time.time() - start_time))
    #     if self.verbose: print("Best action: {} with payoff: {}".format(best_action, best_avg_reward))
    #     return best_action


    def eval_step(self, state, env, player_id):
        return self.step(state, env, player_id)




def roll_out_for_action(env, player_id, action, num_roll_outs, return_dict):
        
    avg_reward = 0.0
        
    # Step forwards
    next_state, next_player_id = env.step(action)
        
    for iteration in range(num_roll_outs):
        depth = 0

        while not env.is_over():
            # Pick a random move using the current env, which will take a random choice of the legal actions, and then look up the index of this, because the get_legal_action function returns the human readable representation of these.
            legal_acts = [env.actions.index(act) for act in env.game.get_legal_actions()]
            random_action = env.actions.index(random.choice(env.game.get_legal_actions()))
            next_state, next_player_id = env.step(random_action)
            depth += 1
        
        payoff = env.get_payoffs()[player_id]
        avg_reward = avg_reward + (payoff - avg_reward) / (iteration + 1)

        for _ in range(depth):
            env.step_back()
    
    return_dict[action] = avg_reward
