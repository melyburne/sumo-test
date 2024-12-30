from util.CustomLoggingCallback import CustomLoggingCallback
from util.utils_model import parse_args_model, get_env
from stable_baselines3 import PPO

def parse_args_ppo():
    prs = parse_args_model("PPO Simple-Intersection")

    return prs.parse_args()


def get_model():
    output_file = "./outputs/2way-single-intersection/ppo"
    out_csv_file = f"{output_file}/sumo"

    args = parse_args_ppo()

    env = get_env(out_csv_file, args)

    model = PPO(
        env=env,
        policy="MlpPolicy",
        learning_rate=args.learning_rate,
        verbose=1,
        device="cpu",
        tensorboard_log=f"{output_file}/tensorboard"
    )
    model.learn(total_timesteps=args.total_timesteps, callback=CustomLoggingCallback())

if __name__ == "__main__":
    get_model()