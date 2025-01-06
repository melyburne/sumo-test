from util.complex_intersection_utils import get_env
from util.evaluate_model import log_tensorboard_evaluate, setup_tensorboard_log_dir
from util.parse_args import parse_args_evaluate, parse_args
from util.random_agent import run_model, log_tensorboard_train
from util.RandomActionModel import RandomActionModel

from stable_baselines3.common.evaluation import evaluate_policy

description_args = "Random Agent Complex-Intersection"
output_file = "./outputs/complex-intersection/random"
out_csv_file = f"{output_file}/sumo"


def train_model():
    env = get_env(out_csv_file, args)
    log_dir = setup_tensorboard_log_dir(f"{output_file}/tensorboard", "random")
    args = parse_args().parse_args()
    episode_rewards = run_model(env, args, args.total_timesteps, args.seconds)
    log_tensorboard_train(log_dir, episode_rewards, args.seconds)


def evaluate_model(ep_length = 200):
    log_dir = setup_tensorboard_log_dir(f"{output_file}/tensorboard", "random_evaluate")
    args = parse_args_evaluate(parse_args(f"{description_args} Evaluate")).parse_args()
    episode_rewards = run_model(args, args.n_eval_episodes * ep_length, ep_length, True)
    log_tensorboard_evaluate(log_dir, episode_rewards)
    return 

if __name__ == "__main__":
    train_model()
    evaluate_model()
