import tensorflow as tf
import os
from .RandomActionModel import RandomActionModel

def log_tensorboard_train(output_file, episode_rewards, seconds):
    """
        Log the rewards in tensorboard.

        :param output_file: Path where output files of tensorboard will be saved.
        :episode_rewards: Array of rewards. Rewards are represented as float.
        :param seconds: Number of simulated seconds on SUMO. The duration in seconds of the simulation.
    """
    writer = tf.summary.create_file_writer(output_file)

    with writer.as_default():
        # Log individual episode rewards and lengths
        for i, reward in enumerate(episode_rewards):
            tf.summary.scalar("rollout/ep_rew_mean", reward, step=(i*seconds) + 1)

        writer.flush()
        writer.close()
