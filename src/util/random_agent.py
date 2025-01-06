import tensorflow as tf
import os
from .RandomActionModel import RandomActionModel

def log_tensorboard_train(output_file, episode_rewards, seconds):
    writer = tf.summary.create_file_writer(output_file)

    with writer.as_default():
        # Log individual episode rewards and lengths
        for i, reward in enumerate(episode_rewards):
            tf.summary.scalar("rollout/ep_rew_mean", reward, step=(i*seconds) + 1)

        writer.flush()
        writer.close()
