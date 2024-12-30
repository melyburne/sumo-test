from util.utils_model import parse_args, get_env
import tensorflow as tf
import os

def parse_args_random():
    prs = parse_args("Random Agent Simple-Intersection")
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


def log_tensorboard(writer_tensorboard, total_reward, step_count, reward, episode_rewards, current_episode):
    # Accumulate reward and count steps
    total_reward += reward
    step_count += 1

    # Add reward to the current episode's reward
    episode_rewards[current_episode] += reward

    # Log step-wise reward
    if (step_count % 100) == 0:
        with writer_tensorboard.as_default():
            tf.summary.scalar("Reward", reward, step=step_count)
            writer_tensorboard.flush()

    return total_reward, step_count, episode_rewards

def log_episode_mean(writer_tensorboard, episode_rewards, current_episode, step_count):
    # Log average reward for the episode
    ep_mean_reward = episode_rewards[current_episode]  # Reward for the current episode
    with writer_tensorboard.as_default():
        tf.summary.scalar("rollout/ep_rew_mean", ep_mean_reward, step=step_count)
        writer_tensorboard.flush()

    return ep_mean_reward

class RandomActionModel:
    """A simple model that selects random actions."""
    def __init__(self, env):
        self.env = env

    def predict(self, _):
        """Returns a random action."""
        action = self.env.action_space.sample()
        return action, None  # Return action and state (None for random actions)

    def train(self, *args, **kwargs):
        """No training required for random actions."""
        pass

    def save(self, path):
        """Save the random model as metadata."""
        with open(path, "w") as f:
            f.write("Random action model - no learnable parameters.")


def train_model():
    output_file = "./outputs/2way-single-intersection/random"
    log_dir = f"{output_file}/tensorboard"
    out_csv_file = f"{output_file}/sumo"

    args = parse_args_random()

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

        total_reward, step_count, episode_rewards = log_tensorboard(writer_tensorboard, total_reward, step_count, reward, episode_rewards, current_episode)

        # Check if the episode is done
        if (step_count % args.seconds) == 0:
            # Log the mean reward for the episode
            log_episode_mean(writer_tensorboard, episode_rewards, current_episode, step_count)
            current_episode += 1
            obs = env.reset()  # Reset for the next episode
            done = False  # Reset done flag

        if step_count == args.total_timesteps:
            done = True

    env.close()
    writer_tensorboard.close()


if __name__ == "__main__":
    train_model()
