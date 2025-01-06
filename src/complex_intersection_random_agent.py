from util.complex_intersection_utils import get_env_raw
from util.evaluate_model import log_tensorboard_evaluate, setup_tensorboard_log_dir
from util.parse_args import parse_args_evaluate, parse_args
from util.random_agent import log_tensorboard_train
from util.RandomActionModel import RandomActionModel

from stable_baselines3.common.evaluation import evaluate_policy

description_args = "Random Agent Complex-Intersection"
output_file = "./outputs/complex-intersection/random"
out_csv_file = f"{output_file}/sumo"


def run_model(env, args, total_timesteps, seconds):
    # Initialize the random action model
    model = RandomActionModel(env)

    obs = env.reset()
    simulationDone = False
    total_reward = 0
    step_count = 0

    current_episode = 0
    episode_rewards = []

    while not simulationDone or env.agents:
        # Use the random model to decide an action
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}

        # Apply the action and get the next observation and reward
        obs, reward, done, truncated, info = env.step(actions)

        # Accumulate reward and count steps
        total_reward += sum(reward.values()) 
        step_count += 1

        # Check if the episode is done (based on `seconds` or `done` condition)
        if (step_count % seconds) == 0:
            # Add the total reward for this episode to the list
            episode_rewards.append(total_reward)

            # Reset for the next episode
            current_episode += 1
            obs = env.reset()
            total_reward = 0  # Reset total reward for the new episode

        # Stop the loop if the total timesteps have been reached
        if step_count == total_timesteps:
            simulationDone = True

    env.close()

    return episode_rewards  # Return the list of rewards

def train_model():
    args = parse_args(f"{description_args} Train").parse_args()
    env = get_env_raw(out_csv_file, args)
    log_dir = setup_tensorboard_log_dir(f"{output_file}/tensorboard", "random")
    episode_rewards = run_model(env, args, args.total_timesteps, args.seconds)
    log_tensorboard_train(log_dir, episode_rewards, args.seconds)


def evaluate_model(ep_length = 200):
    log_dir = setup_tensorboard_log_dir(f"{output_file}/tensorboard", "random_evaluate")
    args = parse_args_evaluate(parse_args(f"{description_args} Evaluate")).parse_args()
    episode_rewards = run_model(args, args.n_eval_episodes * ep_length, ep_length, True)
    log_tensorboard_evaluate(log_dir, episode_rewards)
    return 

if __name__ == "__main__":
    train_model()
    evaluate_model()
