from stable_baselines3.common.evaluation import evaluate_policy

import os
import tensorflow as tf
import numpy as np


def setup_tensorboard_log_dir(base_log_dir, log_dir_add):
    """
        Create a new folder for Tensorbaord logs. Must be unique through addition of numbers.
        
        :param base_log_dir: The base of the log directory.
        :param log_dir_add: What to add to the base directory of the log.
    """
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
    """
        Log episode rewards in output files. 
        
        :param output_file: Path where output files of tensorboard will be saved
        :param episode_rewards: Get array of rewards of a model returns it. Rewards are represented as float.
    """
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

def simple_get_reward(model, args):
    """
        Evaluate a model. Get rewards of a model returns it
        
        :param model: Model to evaluate.
        :param args: ArgumentParser instance with the required arguments.
    """
    episode_rewards, episode_lengths = evaluate_policy(
        model, 
        model.get_env(), 
        n_eval_episodes=args.n_eval_episodes,
        return_episode_rewards=True
    )

    return episode_rewards


def parallel_get_reward(model, args):
    """
        Evaluate a model. Get rewards of a model returns it.
        Workaround because bug doesn't allow the environment to be reset after each episode.
        
        :param model: Model to evaluate.
        :param args: ArgumentParser instance with the required arguments.
    """
    episode_rewards = []
    total_reward = 0
    current_episode = 0

    while current_episode < args.n_eval_episodes:
        rewards, lengths = evaluate_policy(
            model, 
            model.get_env(), 
            n_eval_episodes=1,
        )
        episode_rewards.append(rewards)
        current_episode+=1

    return episode_rewards