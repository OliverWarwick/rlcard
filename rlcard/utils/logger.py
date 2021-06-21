import os
import csv

class Logger(object):
    ''' Logger saves the running results and helps make plots from the results
    '''

    def __init__(self, log_dir, fieldnames=['timestep', 'reward'], csv_name='performance.csv'):
        ''' Initialize the labels, legend and paths of the plot and log file.

        Args:
            log_path (str): The path the log files
        '''
        self.log_dir = log_dir
        self.txt_path = os.path.join(log_dir, 'log.txt')
        self.csv_path = os.path.join(log_dir, csv_name)
        self.fig_path = os.path.join(log_dir, 'fig.png')

        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        self.txt_file = open(self.txt_path, 'w')
        self.csv_file = open(self.csv_path, 'w')
        self.writer = csv.DictWriter(self.csv_file, fieldnames=fieldnames)
        self.writer.writeheader()

    def log(self, text):
        ''' Write the text to log file then print it.
        Args:
            text(string): text to log
        '''
        self.txt_file.write(text+'\n')
        self.txt_file.flush()
        print(text)

    def log_performance(self, timestep, reward):
        ''' Log a point in the curve
        Args:
            timestep (int): the timestep of the current point
            reward (float): the reward of the current point
        '''
        self.writer.writerow({'timestep': timestep, 'reward': reward})
        print('')
        self.log('----------------------------------------')
        self.log('  timestep     |  ' + str(timestep))
        self.log('  reward       |  ' + str(reward))
        self.log('----------------------------------------')
    
    def log_performance_using_algo(self, algo, timestep, reward):
        ''' Log a point in the curve
        Args:
            timestep (int): the timestep of the current point
            algo (str): name of algo used
            reward (float): the reward of the current point
        '''
        self.writer.writerow({'timestep': timestep,'reward': reward})
        print('')
        self.log('----------------------------------------')
        self.log('  algo         |  ' + str(algo))
        self.log('  timestep     |  ' + str(timestep))
        self.log('  reward       |  ' + str(reward))
        self.log('----------------------------------------')
    
    def log_q_value_diffs(self, decision_points, q_value_diffs):
        ''' Log the avg differenece in q values between for a model between
        those given by the network, and the sampled values under the same policies 
        Args:
            decision_points List[Int]: points within the game
            q_value_diffs: List[Float]: differences in q values  
        '''
        for i in range(len(decision_points)):
            self.writer.writerow({'decision point': decision_points[i],'q value difference': q_value_diffs[i]})
            self.log('----------------------------------------')
            self.log('  decision point   |  ' + str(decision_points[i]))
            self.log('  q value diff     |  ' + str(q_value_diffs[i]))
            self.log('----------------------------------------')


    def plot(self, algorithm):
        plot(self.csv_path, self.fig_path, algorithm)

    def duel_plot(self, algorithm1, algorithm2):
        plot_duel(self.csv_path, self.fig_path, algorithm1, algorithm2)
    
    def quad_plot(self, algorithm1, algorithm2, algorithm3, algorithm4):
        plot_quad(self.csv_path, self.fig_path, algorithm1, algorithm2, algorithm3, algorithm4)

    def close_files(self):
        ''' Close the created file objects
        '''
        if self.txt_path is not None:
            self.txt_file.close()
        if self.csv_path is not None:
            self.csv_file.close()

def plot(csv_path, save_path, algorithm):
    ''' Read data from csv file and plot the results
    '''
    import matplotlib.pyplot as plt
    with open(csv_path) as csvfile:
        print(csv_path)
        reader = csv.DictReader(csvfile)
        xs = []
        ys = []
        for row in reader:
            xs.append(int(row['timestep']))
            ys.append(float(row['reward']))
        fig, ax = plt.subplots()
        ax.plot(xs, ys, label=algorithm)
        ax.set(xlabel='timestep', ylabel='reward')
        ax.legend()
        ax.grid()

        save_dir = os.path.dirname(save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        fig.savefig(save_path)

def plot_duel(csv_path, save_path, algorithm1, algorithm2):
    '''
    Reads in the data from a duel stream from CSV. This may be useful for when we have two agents we want to evaluate, or two polices such as in NFSP with Avg and BR.
    '''
    import matplotlib.pyplot as plt
    with open(csv_path) as csvfile:
        print(csv_path)
        reader = csv.DictReader(csvfile)
        xs1, xs2 = [], []
        ys1, ys2 = [], []
        for index, row in enumerate(reader):
            if index % 2 == 0:
                xs1.append(int(row['timestep']))
                ys1.append(float(row['reward']))
            else: 
                xs2.append(int(row['timestep']))
                ys2.append(float(row['reward']))
        fig, ax = plt.subplots()
        ax.plot(xs1, ys1, label=algorithm1)
        ax.plot(xs2, ys2, label=algorithm2)
        ax.set(xlabel='timestep', ylabel='reward')
        ax.legend()
        ax.grid()

        save_dir = os.path.dirname(save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        fig.savefig(save_path)

def plot_quad(csv_path, save_path, algorithm1, algorithm2, algorithm3, algorithm4):
    import matplotlib.pyplot as plt
    with open(csv_path) as csvfile:
        print(csv_path)
        reader = csv.DictReader(csvfile)
        xs1, xs2, xs3, xs4 = [], [], [], []
        ys1, ys2, ys3, ys4 = [], [], [], []
        for index, row in enumerate(reader):
            if index % 4 == 0:
                xs1.append(int(row['timestep']))
                ys1.append(float(row['reward']))
            elif index % 4 == 1:
                xs2.append(int(row['timestep']))
                ys2.append(float(row['reward']))
            elif index % 4 == 2:
                xs3.append(int(row['timestep']))
                ys3.append(float(row['reward']))
            else:
                xs4.append(int(row['timestep']))
                ys4.append(float(row['reward']))
        fig, ax = plt.subplots()
        ax.plot(xs1, ys1, label=algorithm1)
        ax.plot(xs2, ys2, label=algorithm2)
        ax.plot(xs3, ys3, label=algorithm3)
        ax.plot(xs4, ys4, label=algorithm4)
        ax.set(xlabel='timestep', ylabel='rewards')
        ax.legend()
        ax.grid()

        save_dir = os.path.dirname(save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        fig.savefig(save_path)