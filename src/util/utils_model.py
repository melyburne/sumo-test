import argparse
import os
from sumo_rl import SumoEnvironment
import tensorflow as tf
import numpy as np

def parse_args(description):
    prs = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter, description=description
    )
    prs.add_argument("-mingreen", dest="min_green", type=int, default=10, required=False, help="Minimum green time.\n")
    prs.add_argument("-maxgreen", dest="max_green", type=int, default=30, required=False, help="Maximum green time.\n")
    prs.add_argument("-gui", action="store_true", default=False, help="Run with visualization on SUMO.\n")
    prs.add_argument("-s", dest="seconds", type=int, default=1000, required=False, help="Number of simulation seconds.\n")
    prs.add_argument("-tt", dest="total_timesteps", type=int, default=100000, required=False, help="Total timesteps for the model learning.")

    return prs

def parse_args_model(description):
    prs = parse_args(description)
    prs.add_argument("-lr", dest="learning_rate", type=float, default=0.0003, required=False, help="Learning rate of the model.")

    return prs

def parse_args_evaluate(description):
    prs = parse_args(description)
    prs.add_argument("-nee", dest="n_eval_episodes", type=int, default=10, required=False, help="Number of episode to evaluate the agent")

    return prs


def get_file(file_name):
    current_directory = os.getcwd()
    return os.path.join(current_directory, 'data', file_name)


def get_env(out_csv_file, args):
    net_file = get_file('two-way-single-intersection.net.xml')
    route_file = get_file('two-way-single-intersection-gen.rou.xml')
    return SumoEnvironment(
        net_file=net_file,
        route_file=route_file,
        out_csv_name=out_csv_file,
        use_gui=args.gui,
        num_seconds=args.seconds,
        min_green=args.min_green,
        max_green=args.max_green,
        yellow_time=4,
        single_agent=True,
        reward_fn="diff-waiting-time"
    )


def setup_tensorboard_log_dir(base_log_dir, log_dir_add):
    # Set up TensorBoard writer
    if not os.path.exists(base_log_dir):
        os.makedirs(base_log_dir)

    # Find the next available folder name like random_1, random_2, etc.
    counter = 1
    while True:
        log_dir = os.path.join(base_log_dir, f"{log_dir_add}_{counter}")
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            break
        counter += 1

    return log_dir


def log_tensorboard_evaluate(output_file, episode_rewards):
    writer = tf.summary.create_file_writer(output_file)

    with writer.as_default():
        # Log individual episode rewards and lengths
        for i, reward in enumerate(episode_rewards):
            tf.summary.scalar("evaluate/ep_rew_mean", reward, step=i + 1)

        mean_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)

        # Log mean and std of rewards and lengths
        tf.summary.scalar("evaluate/mean_reward", mean_reward, step=1)
        tf.summary.scalar("evaluate/std_reward", std_reward, step=1)
        writer.flush()
        writer.close()
