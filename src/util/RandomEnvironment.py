from util.evaluate_model import log_tensorboard_evaluate, setup_tensorboard_log_dir
from .parse_args import parse_args
from util.random_agent import log_tensorboard_train
from util.RandomActionModel import RandomActionModel

from stable_baselines3.common.evaluation import evaluate_policy

from abc import ABC, abstractmethod

class RandomEnvironment(ABC):

    def __init__(self, output_file, out_csv_file, description_args):
        """
        Initialize the Sumo environment and let the agent perform a random action with it.
        
        :param output_file: Path where output files of tensorboard will be saved
        :param out_csv_file: Path where the output CSV file of SUMO will be saved
        :param description_args: Description of the ArgumentParser
        """
        self.output_file = output_file
        self.out_csv_file = out_csv_file
        self.description_args = description_args

    @abstractmethod
    def get_env(self, args):
        """
            Return the environment for the given arguments.
            The class uses the returned environment to execute the agent.

            :param args: ArgumentParser instance with the required arguments.
        """
        pass

    @abstractmethod
    def run_model(self, env, args, total_timesteps, seconds):
        """
            Defines how to agent interact with the environment.

            :param env: SumoEnvironment for the agent to interact with.
            :param args: ArgumentParser instance with the required arguments.
            :param total_timesteps: The total number of samples (env steps) to train on
            :param seconds: Number of simulated seconds on SUMO. The duration in seconds of the simulation.
        """
        pass

    def train_model(self):
        """
            Agent interact with the environment and the results will be logged as output. 
            This function is used as a baseline to compare models while training.
        """
        args = parse_args(f"{self.description_args} Train")
        env = self.get_env(args)
        log_dir = setup_tensorboard_log_dir(f"{self.output_file}/tensorboard", "random")
        episode_rewards = self.run_model(env, args, args.total_timesteps/5, args.seconds/5)
        log_tensorboard_train(log_dir, episode_rewards, args.seconds)

    def evaluate_model(self):
        """
            Agent interact with the environment and the results will be logged as output. 
            This function is used as a baseline to compare models while evaluation.
        """
        args = parse_args(f"{self.description_args} Evaluate")
        env = self.get_env(args)
        episode_rewards = self.run_model(env, args, (args.n_eval_episodes * args.seconds)/5, (args.seconds/5))
        log_dir = setup_tensorboard_log_dir(f"{self.output_file}/tensorboard", "random_evaluate")
        log_tensorboard_evaluate(f"{log_dir}", episode_rewards)
