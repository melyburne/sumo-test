from util.evaluate_model import log_tensorboard_evaluate, setup_tensorboard_log_dir, get_evaluate_args
from util.parse_args import parse_args_model, parse_args_evaluate, parse_args, parse_args_ppo
from util.get_model import get_ppo_model

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

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
        Initialize the environment with specific parameters.
        
        :param output_file: Path where output files will be saved
        :param out_csv_file: Path to the CSV file used for the environment
        :param description_args: Description of the environment
        :param model_dir: Directory to save the model
        """
        self.output_file = output_file
        self.out_csv_file = out_csv_file
        self.description_args = description_args
        self.model_dir = model_dir


    @abstractmethod
    def get_env(self, args):
        """Return the environment for the given arguments."""
        pass


    def get_ppo_args(self):
        return parse_args_ppo(parse_args_model(parse_args(f"{description_args} Train"))).parse_args()


    def train_model(self, env, args):
        """Train the model."""
        model = get_ppo_model(env, args, self.output_file)
        model.learn(total_timesteps=args.total_timesteps)

        return model

    def save_model(self):
        args = self.get_ppo_args()
        env = self.get_env(args)
        model = self.train_model(env, args)
        model.save(self.model_dir)


    def evaluate_model(self):
        args = get_evaluate_args(self.description_args)
        env = self.get_env(args)
        model = PPO.load(self.model_dir, env=env)
        episode_rewards, episode_lengths = evaluate_policy(
            model, 
            model.get_env(), 
            n_eval_episodes=args.n_eval_episodes,
            return_episode_rewards=True
        )
        log_dir = setup_tensorboard_log_dir(f"{self.output_file}/tensorboard", "ppo_evaluate")
        log_tensorboard_evaluate(f"{log_dir}", episode_rewards)
