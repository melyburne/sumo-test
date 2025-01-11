from .evaluate_model import log_tensorboard_evaluate, setup_tensorboard_log_dir, get_evaluate_args
from .parse_args import parse_args_model, parse_args, parse_args_dqn
from .get_model import get_dqn_model

from stable_baselines3.dqn.dqn import DQN
from stable_baselines3.common.evaluation import evaluate_policy

import os
from abc import ABC, abstractmethod

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable GPU

class DQNEnvironment(ABC):

    def __init__(self, output_file, out_csv_file, description_args, model_dir):
        """
        Initialize the Sumo environment and let the DQN model train, save and evaluate.
        
        :param output_file: Path where output files of tensorboard will be saved
        :param out_csv_file: Path where the output CSV file of SUMO will be saved
        :param description_args: Description of the ArgumentParser
        :param model_dir: Directory to save the model
        """
        self.output_file = output_file
        self.out_csv_file = out_csv_file
        self.description_args = description_args
        self.model_dir = model_dir
    
    
    @abstractmethod
    def get_env(self, args):
        """
            Return the environment for the given arguments.
            The class uses the returned environment to execute the agent.

            :param args: ArgumentParser instance with the required arguments.
        """
        pass


    def get_dqn_args(self):
        """
            Returns the ArgumentParser instance with the arguments required for the agent to work.
        """
        return parse_args_dqn(parse_args_model(parse_args(f"{self.description_args} Train"))).parse_args()


    def train_model(self, env, args):
        """
            Train the model.

            :param env: Gym environment for the agent to interact with.
            :param args: ArgumentParser instance with the required arguments.
        """
        model = get_dqn_model(env, args, self.output_file)
        model.learn(total_timesteps=args.total_timesteps)

        return model

    def save_model(self):
        """
            Train and save a model and log in tensorboard.
            Environment defined in method get_env.
            The constructor parameter model_dir determines the storage location of the model.
        """
        args = self.get_dqn_args()
        env = self.get_env(args)
        model = self.train_model(env, args)
        model.save(self.model_dir)


    def evaluate_model(self):
        """
            Evaluate a model and log in tensorboard.
            Environment defined in method get_env.
            The model used is defined in the constructor parameter model_dir.
        """
        args = get_evaluate_args(self.description_args)
        args.seconds = 200
        env = self.get_env(args)
        model = DQN.load(self.model_dir, env=env)
        episode_rewards, episode_lengths = evaluate_policy(
            model, 
            env, 
            n_eval_episodes=args.n_eval_episodes,
            return_episode_rewards=True
        )
        log_dir = setup_tensorboard_log_dir(f"{self.output_file}/tensorboard", "dqn_evaluate")
        log_tensorboard_evaluate(f"{log_dir}", episode_rewards)