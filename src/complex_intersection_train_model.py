from complex_intersection_dqn import save_model as dqn_save_model
from complex_intersection_ppo import save_model as ppo_save_model
from complex_intersection_random_agent import train_model as random_agent_train_model

if __name__ == "__main__":
    # dqn_save_model()
    # ppo_save_model()
    # To compare rewards
    random_agent_train_model()