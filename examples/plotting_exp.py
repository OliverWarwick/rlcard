import rl_plotter
from rl_plotter.plot_utils import plot_results
import pandas as pd
import os
import matplotlib.pyplot as plt

def load_csv_data_for_model(dir_name, file_name):

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


def plot_avg_for_models(df, show=True, save=True, save_dir=None):
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

    colors = ['blue', 'red', 'green']
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

def plot_for_models(df):

    ''' 
    For use with only one training set
    '''
    legend, = plt.plot(df.timestep, df.reward, color=colors[index])
    plt.xlabel("Timestep")
    plt.ylabel("Avg Reward")
    plt.legend(df.model.unique())
    plt.title("Eval Reward vs Time")
    plt.show()



if __name__ == '__main__':
    dqn = load_csv_data_for_model(dir_name = ".ow_model/experiments/nano_ofcp_dqn_vs_random_training_run/", file_name = "dqn.csv")
    nfsp_br = load_csv_data_for_model(dir_name = '.ow_model/experiments/nano_ofcp_nfsp_result_exper/', file_name = "nfsp_br.csv")
    nfsp_ap = load_csv_data_for_model(dir_name = '.ow_model/experiments/nano_ofcp_nfsp_result_exper/', file_name = "nfsp_ap.csv")
    plot_avg_for_models(pd.concat([dqn, nfsp_ap, nfsp_br]), show=True, save=True, save_dir='.ow_model/experiments/nano_ofcp_nfsp_result_exper/')
