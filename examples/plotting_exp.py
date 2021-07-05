import rl_plotter
from rl_plotter.plot_utils import plot_results
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np

def load_csv_data_for_timeseries(dir_name, file_name):

    # We should see that the filename of the csv is the type of model e.g: dqn.csv
    # We can just grab the first part of this and use as the model name.
    model_type = file_name[:-4].upper()
    print(model_type)

    dataframes = []

    for root, dirs, files in os.walk(dir_name, topdown=False):
        for name in files:
            if(name == file_name):
                full_path = os.path.join(root, name)

                # Load the dataframe
                frame = pd.read_csv(filepath_or_buffer=full_path)
                dataframes.append(frame)

    # Once we have the dataframes we can then use these     
    df = pd.concat(dataframes)
    df.sort_values('timestep', inplace=True)
    df.reset_index(inplace=True)          
    df['model'] = model_type
    return df


def plot_one_std_timeseries_for_models(df, show=True, save=True, save_dir=None):
    ''' 
    For use when more than one training run per model. 
    We take the avg and std and use this as the plots 


    args:
        df: Dataframe with columns, timestep, reward (in eval env) and model
            e.g:    index  timestep  reward Model
            0      0         6   0.009   DQN
            1      0         6   0.009   DQN
            2      1      6006   0.067   DQN
            ...
            500    0         6   0.076   NFSP_BR
    '''

    colors = ['red', 'blue', 'green']
    model_names = df.model.unique() 
    # Firstly, group the data by timestamp and model.

    grouped_data = df.groupby(['model'])
    for index, model_name in enumerate(model_names):

        model_wise_data = grouped_data.get_group(model_name)
        timestep_grouped_data = model_wise_data.groupby(['timestep'], as_index=False)

        mean_data = timestep_grouped_data.mean()
        std_data = timestep_grouped_data.std()

        mean_data['lower_std_reward'] = mean_data.reward - std_data.reward
        mean_data['upper_std_reward'] = mean_data.reward + std_data.reward

        print(mean_data)
        # mean_data['reward'].plot(legend=model_name, color=colors[index])
        # plt.show()
        legend, = plt.plot(mean_data.timestep, mean_data.reward, color=colors[index])
        plt.fill_between(mean_data.timestep, mean_data.lower_std_reward, mean_data.upper_std_reward, color=colors[index], alpha=.2)

        # for i, model in enumerate(mean_data.model):
        #     specific_model_data = mean_data[mean_data[model] == model]
        #     print(specific_model_data)
        #     legend, = plt.plot(specific_model_data.timestep, specific_model_data.reward, color=colors[index])
        #     plt.fill_between(specific_model_data.timestep, specific_model_data.lower_std_reward, specific_model_data.upper_std_reward, color=colors[index], alpha=.2)
    plt.xlabel("Timestep")
    plt.ylabel("Avg Reward")
    plt.legend(model_names)
    plt.title("Avg Eval Reward vs Time")
    plt.savefig(save_dir + 'line_graph.png')
    plt.show()

def plot_single_timeseries_for_model(df):

    ''' 
    For use with only one training set
    '''
    legend, = plt.plot(df.timestep, df.reward, color=colors[index])
    plt.xlabel("Timestep")
    plt.ylabel("Avg Reward")
    plt.legend(df.model.unique())
    plt.title("Eval Reward vs Time")
    plt.show()



def load_q_value_diff_data(dir_name, file_name):

    data = [[], [], []]

    for root, dirs, files in os.walk(dir_name, topdown=False):
        for name in files:
            if(name == file_name):
                full_path = os.path.join(root, name)

                # Load the dataframe
                frame = pd.read_csv(filepath_or_buffer=full_path)
                print(frame)
                for index, row in frame.iterrows():
                    data[index].append(row[1])
    return data


def plot_box_whisker_for_q_value_diff(data_a, model_name_a, data_b, model_name_b, save_dir):

    ticks = ['First Deal', 'Second Deal', 'Third Deal']

    def set_box_color(bp, color):
        plt.setp(bp['boxes'], color=color)
        plt.setp(bp['whiskers'], color=color)
        plt.setp(bp['caps'], color=color)
        plt.setp(bp['medians'], color=color)

    plt.figure()

    bpl = plt.boxplot(data_a, positions=np.array(range(len(data_a)))*2.0-0.4, sym='', widths=0.6)
    bpr = plt.boxplot(data_b, positions=np.array(range(len(data_b)))*2.0+0.4, sym='', widths=0.6)
    set_box_color(bpl, '#D7191C') # colors are from http://colorbrewer2.org/
    set_box_color(bpr, '#2C7BB6')

    # draw temporary red and blue lines and use them to create a legend
    plt.plot([], c='#D7191C', label=model_name_a)
    plt.plot([], c='#2C7BB6', label=model_name_b)
    plt.legend()

    plt.xticks(range(0, len(ticks) * 2, 2), ticks)
    plt.xlim(-2, len(ticks)*2)
    plt.ylim(-0.8, 0.8)
    plt.ylabel("Differnece in Q Value")
    plt.title("Difference in Q Value Estimation")
    plt.tight_layout()
    plt.show()
    plt.savefig('boxcompare.png')



def plot_states_prop(data):

    state_data = data[['episode', 'states']]
    prop_data = data[['episode', 'prop']]

    # Plot on the same graoh, the number of states encounting.
    # And also the propotion of the latest states which are new.

    # Create a mean of the data, and then a std coloum for each.
    # Firstly, group the data by timestamp and model.

    state_data_mean = state_data.groupby('episode').mean()
    state_data_std = state_data.groupby('episode').std()
    state_data_mean['upperbound'] = state_data_mean['states'] + state_data_std['states']
    state_data_mean['lowerbound'] = state_data_mean['states'] - state_data_std['states']

    prop_data_mean = prop_data.groupby('episode').mean()
    prop_data_std = prop_data.groupby('episode').std()
    prop_data_mean['upperbound'] = prop_data_mean['prop'] + prop_data_std['prop']
    prop_data_mean['lowerbound'] = prop_data_mean['prop'] - prop_data_std['prop']

    print(state_data_mean)
    print(prop_data_mean)

    # create figure and axis objects with subplots()
    fig,ax = plt.subplots()
    # make a plot
    state_data_mean['states'].plot(color="red", marker="o", ax=ax)
    # ax.set_yscale('log')
    ax.fill_between(state_data_mean.index, state_data_mean.lowerbound, state_data_mean.upperbound, color="red", alpha=.2)
    # set x-axis label
    ax.set_xlabel("Episodes",fontsize=14)
    # set y-axis label
    ax.set_ylabel("Number of Distinct States",color="red",fontsize=14)

    # twin object for two different y-axis on the sample plot
    ax2=ax.twinx()
    # make a plot with different y-axis using second axis object
    prop_data_mean['prop'].plot(color="blue",marker="o", ax=ax2)
    ax2.fill_between(prop_data_mean.index, prop_data_mean.lowerbound, prop_data_mean.upperbound, color="blue", alpha=.2)
    ax2.set_ylabel("Proportion of New States Previously Unvisited", color="blue", fontsize=14)
    plt.title("Nano OFCP State Space Growth under Exploration")
    plt.show()
    # save the plot as a file
    fig.savefig('states_vs_prob_new.jpg',
                format='png',
                dpi=500,
                bbox_inches='tight')







def timeseries_runner():
    dqn = load_csv_data_for_timeseries(dir_name = "ow_model/experiments/nano_ofcp_dqn_vs_random_training_run/", file_name = "dqn.csv")
    dqn_with_neg = load_csv_data_for_timeseries(dir_name = "ow_model/experiments/nano_ofcp_dqn_neg/", file_name = "dqn_adj_illegal.csv")
    # nfsp_br = load_csv_data_for_model(dir_name = '.ow_model/experiments/nano_ofcp_nfsp_result_exper/', file_name = "nfsp_br.csv")
    # nfsp_ap = load_csv_data_for_model(dir_name = '.ow_model/experiments/nano_ofcp_nfsp_result_exper/', file_name = "nfsp_ap.csv")
    plot_one_std_timeseries_for_models(pd.concat([dqn, dqn_with_neg]), show=True, save=True, save_dir='ow_model/experiments/nano_ofcp_nfsp_result_exper/')

def box_whisker_runner():
    dqn = load_q_value_diff_data(dir_name = "ow_model/experiments/nano_ofcp_dqn_vs_random_training_run/", file_name = "q_values_diffs_v2.csv")
    dqn_neg_reward = load_q_value_diff_data(dir_name = "ow_model/experiments/nano_ofcp_dqn_neg/", file_name = "q_values_diffs_v2.csv")
    print(dqn)
    print(dqn_neg_reward)
    plot_box_whisker_for_q_value_diff(dqn, "DQN", dqn_neg_reward, "DQN_ADJ_ILLEGAL", ".ow_model/experiments/nano_ofcp_dqn_vs_random_training_run/")
    


def state_prop_runner():
    data = pd.read_csv("state_data.csv")
    non_zero_rows = data[data['episode'] >= 500]
    plot_states_prop(non_zero_rows)





if __name__ == '__main__':
    state_prop_runner()
    # box_whisker_runner()
