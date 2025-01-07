from util.evaluate_model import log_tensorboard_evaluate, setup_tensorboard_log_dir, get_evaluate_args
from util.parse_args import parse_args
from util.random_agent import log_tensorboard_train
from util.RandomActionModel import RandomActionModel

from stable_baselines3.common.evaluation import evaluate_policy

from abc import ABC, abstractmethod

class RandomEnvironment(ABC):

    def __init__(self, output_file, out_csv_file, description_args):
        """
        Initialize the environment with specific parameters.
        
        :param output_file: Path where output files will be saved
        :param out_csv_file: Path to the CSV file used for the environment
        :param description_args: Description of the environment
        :param model_dir: Directory to save the model
        """
        self.output_file = output_file
        self.out_csv_file = out_csv_file
        self.description_args = description_args

    @abstractmethod
    def get_env(self, args):
        """Return the environment for the given arguments."""
        pass

    def get_random_args(self):
        return parse_args(f"{self.description_args} Train").parse_args()

    @abstractmethod
    def run_model(self, env, args, total_timesteps, seconds):
        pass

    def train_model(self):
        """Train the model."""
        args = self.get_random_args()
        env = self.get_env(args)
        log_dir = setup_tensorboard_log_dir(f"{self.output_file}/tensorboard", "random")
        episode_rewards = self.run_model(env, args, args.total_timesteps, args.seconds)
        log_tensorboard_train(log_dir, episode_rewards, args.seconds)

    def evaluate_model(self, ep_length = 200):
        args = get_evaluate_args(self.description_args)
        env = self.get_env(args)
        episode_rewards = self.run_model(env, args, args.n_eval_episodes * ep_length, ep_length)
        log_dir = setup_tensorboard_log_dir(f"{self.output_file}/tensorboard", "random_evaluate")
        log_tensorboard_evaluate(f"{log_dir}", episode_rewards)
