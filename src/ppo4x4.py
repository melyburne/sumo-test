from util.utils_model import parse_args_model, log_tensorboard_evaluate, parse_args_evaluate, setup_tensorboard_log_dir, get_file
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
import tensorflow as tf
import sumo_rl
from stable_baselines3.common.vec_env import VecMonitor
import supersuit as ss
import types
from typing import Callable

output_file = "./outputs/4x4/ppo"
out_csv_file = f"{output_file}/sumo"
description_args = "PPO 4x4-Grid"

def parse_args_ppo():
    prs = parse_args_model(f"{description_args} Train")

    return prs.parse_args()

def get_env(out_csv_file, args):
    net_file = get_file('4x4.net.xml')
    route_file = get_file('4x4c1c2c1c2.rou.xml')
    
    env = sumo_rl.parallel_env(
        net_file=net_file,
        route_file=route_file,
        out_csv_name=out_csv_file,
        use_gui=args.gui,
        num_seconds=args.seconds,
        reward_fn="diff-waiting-time",
    )
    env.unwrapped.render_mode = 'None'
    return env

def get_model():
    args = parse_args_ppo()

    env = get_env(out_csv_file, args)

    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, 2, num_cpus=1, base_class="stable_baselines3")
    env = VecMonitor(env)

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


def save_model(name = "models/ppo_4x4"):
    model = get_model()
    model.save(name)


def evaluate_model(name = "models/ppo_4x4"):
    args = parse_args_evaluate(f"{description_args} Evaluate").parse_args()
    env = get_env(out_csv_file, args)
    model = PPO.load(name, env=env)
    episode_rewards, episode_lengths = evaluate_policy(
        model, 
        model.get_env(), 
        n_eval_episodes=10,
        return_episode_rewards=True
    )
    log_dir = setup_tensorboard_log_dir(f"{output_file}/tensorboard", "ppo_4x4_evaluate")
    log_tensorboard_evaluate(f"{log_dir}", episode_rewards)


if __name__ == "__main__":
    save_model()
    evaluate_model()