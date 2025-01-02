from util.utils_model import parse_args, get_env, parse_args_evaluate
import tensorflow as tf
import os
from stable_baselines3.common.evaluation import evaluate_policy
from util.RandomActionModel import RandomActionModel

description_args = "Random Agent Simple-Intersection"

def parse_args_random():
    prs = parse_args(description_args)
    return prs.parse_args()


def setup_tensorboard(base_log_dir):
    # Set up TensorBoard writer
    if not os.path.exists(base_log_dir):
        os.makedirs(base_log_dir)

    # Find the next available folder name like random_1, random_2, etc.
    counter = 1
    while True:
        log_dir = os.path.join(base_log_dir, f"random_{counter}")
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            break
        counter += 1

    return tf.summary.create_file_writer(log_dir)


def log_tensorboard(writer_tensorboard, step_count, reward):
    # Log step-wise reward
    if (step_count % 100) == 0:
        with writer_tensorboard.as_default():
            tf.summary.scalar("rollout/reward", reward, step=step_count)
            writer_tensorboard.flush()


def log_episode_mean(writer_tensorboard, episode_rewards, current_episode, step_count, scalar_name):
    # Log average reward for the episode
    ep_mean_reward = episode_rewards[current_episode]  # Reward for the current episode
    with writer_tensorboard.as_default():
        tf.summary.scalar(scalar_name, ep_mean_reward, step=step_count)
        writer_tensorboard.flush()


def run_model(args, total_timesteps, seconds, evaluate = False):
    output_file = "./outputs/2way-single-intersection/random"
    log_dir = f"{output_file}/tensorboard"
    out_csv_file = f"{output_file}/sumo"

    env = get_env(out_csv_file, args)
    writer_tensorboard = setup_tensorboard(log_dir)

    # Initialize the random action model
    model = RandomActionModel(env)

    obs = env.reset()
    done = False
    total_reward = 0
    step_count = 0

    current_episode = 0
    episode_rewards = {}

    while not done:
        if current_episode not in episode_rewards:
            episode_rewards[current_episode] = 0

        # Use the random model to decide an action
        action, _ = model.predict(obs)

        # Apply the action and get the next observation and reward
        obs, reward, done, truncated, info = env.step(action)

        # Accumulate reward and count steps
        total_reward += reward
        step_count += 1

        # Add reward to the current episode's reward
        episode_rewards[current_episode] += reward

        if not evaluate:
            log_tensorboard(writer_tensorboard, step_count, reward)

        # Check if the episode is done
        if (step_count % seconds) == 0:
            # Log the mean reward for the episode
            if evaluate:
                log_episode_mean(writer_tensorboard, episode_rewards, current_episode, current_episode, "evaluate/ep_rew_mean")
            else:
                log_episode_mean(writer_tensorboard, episode_rewards, current_episode, step_count, "rollout/ep_rew_mean")
            current_episode += 1
            obs = env.reset()  # Reset for the next episode
            done = False  # Reset done flag

        if step_count == total_timesteps:
            done = True

    env.close()
    writer_tensorboard.close()

def train_model():
    args = parse_args_random()
    run_model(args, args.total_timesteps, args.seconds)


def evaluate_model():
    args = parse_args_evaluate(f"{description_args} Evaluate").parse_args()
    ep_length = 200
    run_model(args, args.n_eval_episodes * ep_length, ep_length, True)
    return 

if __name__ == "__main__":
    train_model()
    evaluate_model()
