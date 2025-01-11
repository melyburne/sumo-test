from stable_baselines3 import PPO
from stable_baselines3 import DQN

def get_ppo_model(env, args, output_file):
    """
        Returns a PPO model.

        :param env: Gym environment for the agent to interact with.
        :param args: ArgumentParser instance with the required arguments.
        :param output_file: Path where output files of tensorboard will be saved
    """
    return PPO(
        env=env,
        policy="MlpPolicy",
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        batch_size=args.batch_size,
        n_steps=args.n_steps,
        clip_range=args.clip_range,
        verbose=1,
        device="cpu",
        tensorboard_log=f"{output_file}/tensorboard"
    )

def get_dqn_model(env, args, output_file):
    """
        Returns a DQN model.

        :param env: Gym environment for the agent to interact with.
        :param args: ArgumentParser instance with the required arguments.
        :param output_file: Path where output files of tensorboard will be saved
    """
    return DQN(
        env=env,
        policy="MlpPolicy",
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        batch_size=args.batch_size,
        learning_starts=args.learning_starts,
        train_freq=(args.train_freq, "step"),
        target_update_interval=args.target_update_interval,
        exploration_initial_eps=args.exploration_initial_eps,
        exploration_final_eps=args.exploration_final_eps,
        verbose=2,
        tensorboard_log=f"{output_file}/tensorboard",
    )