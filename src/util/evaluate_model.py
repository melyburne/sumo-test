import os
import tensorflow as tf
import numpy as np


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