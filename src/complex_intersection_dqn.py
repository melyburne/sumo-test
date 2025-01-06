from util.evaluate_model import log_tensorboard_evaluate, setup_tensorboard_log_dir
from util.complex_intersection_utils import get_env
from util.parse_args import parse_args_model, parse_args_evaluate, parse_args, parse_args_dqn
from util.get_model import get_dqn_model

from stable_baselines3.dqn.dqn import DQN
from stable_baselines3.common.evaluation import evaluate_policy

import tensorflow as tf

output_file = "./outputs/complex-intersection/dqn"
out_csv_file = f"{output_file}/sumo"
description_args = "DQN Complex-Intersection"
model_dir = "models/dqn_complex-intersection"


def train_model():
    args = parse_args_dqn(parse_args_model(parse_args(f"{description_args} Train"))).parse_args()

    env = get_env(out_csv_file, args)

    model = get_dqn_model(env, args, output_file)
    model.learn(total_timesteps=args.total_timesteps)

    return model


def save_model(name = model_dir):
    model = train_model()
    model.save(name)


def evaluate_model(name = model_dir):
    args = parse_args_evaluate(parse_args(f"{description_args} Evaluate")).parse_args()
    env = get_env(out_csv_file, args)
    model = DQN.load(name, env=env)
    episode_rewards, episode_lengths = evaluate_policy(
        model, 
        model.get_env(), 
        n_eval_episodes=args.n_eval_episodes,
        return_episode_rewards=True
    )
    log_dir = setup_tensorboard_log_dir(f"{output_file}/tensorboard", "dqn_evaluate")
    log_tensorboard_evaluate(f"{log_dir}", episode_rewards)


if __name__ == "__main__":
    save_model()
    evaluate_model()