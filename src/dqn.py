from util.CustomLoggingCallback import CustomLoggingCallback
from util.utils_model import parse_args_model, get_env, log_tensorboard_evaluate, parse_args_evaluate
from stable_baselines3.dqn.dqn import DQN
from stable_baselines3.common.evaluation import evaluate_policy
import tensorflow as tf

output_file = "./outputs/2way-single-intersection/dqn"
out_csv_file = f"{output_file}/sumo"
description_args = "DQN Simple-Intersection"


def parse_args_dqn():
    prs = parse_args_model(f"{description_args} Train")
    prs.add_argument("-ls", dest="learning_starts", type=int, default=0, required=False, help="Number of steps before learning starts in the DQN model.")
    prs.add_argument("-tf", dest="train_freq", type=int, default=1, required=False, help="Training frequency of the DQN model.")
    prs.add_argument("-tui", dest="target_update_interval", type=int, default=500, required=False, help="Interval for updating the target network in the DQN model.")
    prs.add_argument("-eie", dest="exploration_initial_eps", type=float, default=0.05, required=False, help="Initial exploration epsilon for the DQN model.")
    prs.add_argument("-efe", dest="exploration_final_eps", type=float, default=0.01, required=False, help="Final exploration epsilon for the DQN model.")

    return prs.parse_args()


def get_model():
    args = parse_args_dqn()

    env = get_env(out_csv_file, args)

    model = DQN(
        env=env,
        policy="MlpPolicy",
        learning_rate=args.learning_rate,
        learning_starts=args.learning_starts,
        train_freq=args.train_freq,
        target_update_interval=args.target_update_interval,
        exploration_initial_eps=args.exploration_initial_eps,
        exploration_final_eps=args.exploration_final_eps,
        verbose=1,
        tensorboard_log=f"{output_file}/tensorboard",
    )
    model.learn(total_timesteps=args.total_timesteps, callback=CustomLoggingCallback())

    return model

def save_model(name = "models/dqn_simple-intersection"):
    model = get_model()
    model.save(name)

def evaluate_model(name = "models/dqn_simple-intersection"):
    args = parse_args_evaluate(f"{description_args} Evaluate").parse_args()
    env = get_env(out_csv_file, args)
    model = DQN.load(name, env=env)
    episode_rewards, episode_lengths = evaluate_policy(
        model, 
        model.get_env(), 
        n_eval_episodes=args.n_eval_episodes,
        return_episode_rewards=True
    )
    log_tensorboard_evaluate(f"{output_file}/tensorboard", episode_rewards, episode_lengths)

if __name__ == "__main__":
    save_model()
    evaluate_model()