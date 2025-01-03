from util.utils_model import parse_args_model, get_env, log_tensorboard_evaluate, parse_args_evaluate, setup_tensorboard_log_dir
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
import tensorflow as tf

output_file = "./outputs/2way-single-intersection/ppo"
out_csv_file = f"{output_file}/sumo"
description_args = "PPO Simple-Intersection"

def parse_args_ppo():
    prs = parse_args_model(f"{description_args} Train")

    return prs.parse_args()


def get_model():
    args = parse_args_ppo()

    env = get_env(out_csv_file, args)

    model = PPO(
        env=env,
        policy="MlpPolicy",
        learning_rate=args.learning_rate,
        verbose=1,
        device="cpu",
        tensorboard_log=f"{output_file}/tensorboard"
    )
    model.learn(total_timesteps=args.total_timesteps)
    return model

def save_model(name = "models/ppo_simple-intersection"):
    model = get_model()
    model.save(name)

def evaluate_model(name = "models/ppo_simple-intersection"):
    args = parse_args_evaluate(f"{description_args} Evaluate").parse_args()
    env = get_env(out_csv_file, args)
    model = PPO.load(name, env=env)
    episode_rewards, episode_lengths = evaluate_policy(
        model, 
        model.get_env(), 
        n_eval_episodes=10,
        return_episode_rewards=True
    )
    log_dir = setup_tensorboard_log_dir(f"{output_file}/tensorboard", "ppo_evaluate")
    log_tensorboard_evaluate(f"{log_dir}", episode_rewards)

if __name__ == "__main__":
    save_model()
    evaluate_model()