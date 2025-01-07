import argparse

def parse_args(description):
    prs = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter, description=description
    )
    # SUMO Variables
    prs.add_argument("-mingreen", dest="min_green", type=int, default=10, required=False, help="Minimum green time.\n")
    prs.add_argument("-maxgreen", dest="max_green", type=int, default=30, required=False, help="Maximum green time.\n")
    prs.add_argument("-gui", action="store_true", default=False, help="Run with visualization on SUMO.\n")
    prs.add_argument("-s", dest="seconds", type=int, default=1000, required=False, help="Number of simulation seconds.\n")

    # Variable for model training
    prs.add_argument("-tt", dest="total_timesteps", type=int, default=100000, required=False, help="Total timesteps for the model learning.")

    return prs

def parse_args_model(prs):
    prs.add_argument("-lr", dest="learning_rate", type=float, default=0.0003, required=False, help="Learning rate of the model.")
    prs.add_argument("-g", dest="gamma", type=float, default=0.99, required=False, help="Discount factor of the model.")
    prs.add_argument("-bs", dest="batch_size", type=int, default=64, required=False, help="Minibatch size for each gradient update of the model.")
    return prs

def parse_args_evaluate(prs):
    prs.add_argument("-nee", dest="n_eval_episodes", type=int, default=10, required=False, help="Number of episode to evaluate the agent")
    return prs

def parse_args_ppo(prs):
    prs.add_argument("-ns", dest="n_steps", type=int, default=256, required=False, help="The number of steps to run for each environment per update in the PPO model.")
    prs.add_argument("-gl", dest="gae_lambda", type=float, default=0.95, required=False, help="Factor for trade-off of bias vs variance for Generalized Advantage Estimator in the PPO model.")
    prs.add_argument("-cr", dest="clip_range", type=float, default=0.95, required=False, help="Clipping parameter in the PPO model.")
    return prs

def parse_args_dqn(prs):
    prs.add_argument("-ls", dest="learning_starts", type=int, default=0, required=False, help="Number of steps before learning starts in the DQN model.")
    prs.add_argument("-tf", dest="train_freq", type=int, default=1, required=False, help="Training frequency of the DQN model.")
    prs.add_argument("-tui", dest="target_update_interval", type=int, default=500, required=False, help="Interval for updating the target network in the DQN model.")
    prs.add_argument("-eie", dest="exploration_initial_eps", type=float, default=0.05, required=False, help="Initial exploration epsilon for the DQN model.")
    prs.add_argument("-efe", dest="exploration_final_eps", type=float, default=0.01, required=False, help="Final exploration epsilon for the DQN model.")
    return prs