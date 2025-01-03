from util.utils_model import parse_args, get_env, parse_args_evaluate, log_tensorboard_evaluate, setup_tensorboard_log_dir
import tensorflow as tf
import os
from stable_baselines3.common.evaluation import evaluate_policy
from util.RandomActionModel import RandomActionModel

description_args = "Random Agent Simple-Intersection"
output_file = "./outputs/2way-single-intersection/random"
out_csv_file = f"{output_file}/sumo"

def parse_args_random():
    prs = parse_args(description_args)
    return prs.parse_args()


def log_tensorboard_train(output_file, episode_rewards, seconds):
    writer = tf.summary.create_file_writer(output_file)

    with writer.as_default():
        # Log individual episode rewards and lengths
        for i, reward in enumerate(episode_rewards):
            tf.summary.scalar("rollout/ep_rew_mean", reward, step=(i*seconds) + 1)

        writer.flush()
        writer.close()


def run_model(args, total_timesteps, seconds, evaluate=False):
    env = get_env(out_csv_file, args)

    # Initialize the random action model
    model = RandomActionModel(env)

    obs = env.reset()
    done = False
    total_reward = 0
    step_count = 0

    current_episode = 0
    episode_rewards = []

    while not done:
        # Use the random model to decide an action
        action, _ = model.predict(obs)

        # Apply the action and get the next observation and reward
        obs, reward, done, truncated, info = env.step(action)

        # Accumulate reward and count steps
        total_reward += reward
        step_count += 1

        # Check if the episode is done (based on `seconds` or `done` condition)
        if (step_count % seconds) == 0 or done:
            # Add the total reward for this episode to the list
            episode_rewards.append(total_reward)

            # Reset for the next episode
            current_episode += 1
            obs = env.reset()
            total_reward = 0  # Reset total reward for the new episode
            done = False  # Reset done flag

        # Stop the loop if the total timesteps have been reached
        if step_count == total_timesteps:
            done = True

    env.close()

    return episode_rewards  # Return the list of rewards

def train_model():
    log_dir = setup_tensorboard_log_dir(f"{output_file}/tensorboard", "random")
    args = parse_args_random()
    episode_rewards = run_model(args, args.total_timesteps, args.seconds)
    log_tensorboard_train(log_dir, episode_rewards, args.seconds)


def evaluate_model():
    log_dir = setup_tensorboard_log_dir(f"{output_file}/tensorboard", "random_evaluate")
    args = parse_args_evaluate(f"{description_args} Evaluate").parse_args()
    ep_length = 200
    episode_rewards = run_model(args, args.n_eval_episodes * ep_length, ep_length, True)
    log_tensorboard_evaluate(log_dir, episode_rewards)
    return 

if __name__ == "__main__":
    train_model()
    evaluate_model()
