from util.utils_model import parse_args, get_file, get_env
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


def log_tensorboard(writer_tensorboard, total_reward, step_count, reward):
    # Accumulate reward and count steps
    total_reward += reward
    step_count += 1

    # Log step-wise data in real-time
    if (step_count % 100) == 0:
        with writer_tensorboard.as_default():
            print(reward)
            tf.summary.scalar("Reward", reward, step=step_count)
            writer_tensorboard.flush()

    return total_reward, step_count


def get_model():
    output_file = "./outputs/2way-single-intersection/random"
    log_dir = f"{output_file}/tensorboard"
    out_csv_file = f"{output_file}/sumo"

    args = parse_args_random()

    env = get_env(out_csv_file, args)

    writer_tensorboard = setup_tensorboard(log_dir)

    env.reset()
    done = False
    total_reward = 0
    step_count = 0

    while not done:
        # Select a random action from the action space
        action = env.action_space.sample()
        # Apply the action and get the next observation and reward
        obs, reward, done, truncated, info = env.step(action)

        total_reward, step_count = log_tensorboard(writer_tensorboard, total_reward, step_count, reward)

        if step_count == args.total_timesteps:
            done = True

    env.close()
    writer_tensorboard.close()


if __name__ == "__main__":
    get_model()
