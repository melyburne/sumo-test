from util.evaluate_model import log_tensorboard_evaluate, setup_tensorboard_log_dir, simple_get_reward
from .parse_args import parse_args
from util.get_model import get_ppo_model

from stable_baselines3 import PPO

import os
from abc import ABC, abstractmethod

output_file = "./outputs/complex-intersection/ppo"
out_csv_file = f"{output_file}/sumo"
description_args = "PPO Complex Intersection"
model_dir = "models/ppo_complex-intersection"

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable GPU

class PPOEnvironment(ABC):

    def __init__(self, output_file, out_csv_file, description_args, model_dir):
        """
        Initialize the Sumo environment and let the PPO model train, save and evaluate.
        
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


    def train_model(self, env, args):
        """
            Train the model.

            :param env: Gym environment for the agent to interact with.
            :param args: ArgumentParser instance with the required arguments.
        """
        model = get_ppo_model(env, args, self.output_file)
        model.learn(total_timesteps=args.total_timesteps)

        return model

    def save_model(self):
        """
            Train and save a model and log in tensorboard.
            Environment defined in method get_env.
            The constructor parameter model_dir determines the storage location of the model.
        """
        args = parse_args(f"{self.description_args} Train")
        env = self.get_env(args)
        model = self.train_model(env, args)
        model.save(self.model_dir)


    def evaluate_model(self):
        """
            Evaluate a model and log in tensorboard.
            Environment defined in method get_env.
            The model used is defined in the constructor parameter model_dir.
        """
        args = parse_args(f"{self.description_args} Evaluate")
        env = self.get_env(args)
        model = PPO.load(self.model_dir, env=env)
        episode_rewards = self.evaluate_model_get_reward(model, args)
        log_dir = setup_tensorboard_log_dir(f"{self.output_file}/tensorboard", "ppo_evaluate")
        log_tensorboard_evaluate(f"{log_dir}", episode_rewards)

    def evaluate_model_get_reward(self, model, args):
        """
            Evaluate a model. Get array of rewards of a model returns it. Rewards are represented as float.
            
            :param model: Model to evaluate.
            :param args: ArgumentParser instance with the required arguments.
        """
        return simple_get_reward(model, args)
