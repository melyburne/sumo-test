from util.CustomLoggingCallback import CustomLoggingCallback
from util.utils_model import parse_args_model, get_file, get_env
from stable_baselines3.dqn.dqn import DQN

def parse_args_dqn():
    prs = parse_args_model("DQN Simple-Intersection")
    prs.add_argument("-ls", dest="learning_starts", type=int, default=0, required=False, help="Number of steps before learning starts in the DQN model.")
    prs.add_argument("-tf", dest="train_freq", type=int, default=1, required=False, help="Training frequency of the DQN model.")
    prs.add_argument("-tui", dest="target_update_interval", type=int, default=500, required=False, help="Interval for updating the target network in the DQN model.")
    prs.add_argument("-eie", dest="exploration_initial_eps", type=float, default=0.05, required=False, help="Initial exploration epsilon for the DQN model.")
    prs.add_argument("-efe", dest="exploration_final_eps", type=float, default=0.01, required=False, help="Final exploration epsilon for the DQN model.")

    return prs.parse_args()


def get_model():
    output_file = "./outputs/2way-single-intersection/dqn"
    out_csv_file = f"{output_file}/sumo"

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

if __name__ == "__main__":
    get_model()