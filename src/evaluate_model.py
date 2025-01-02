from dqn import evaluate_model as dqn_evaluate_model
from ppo import evaluate_model as ppo_evaluate_model
from random_agent import evaluate_model as random_agent_evaluate_model

if __name__ == "__main__":
    dqn_evaluate_model()
    ppo_evaluate_model()
    # To compare rewards
    random_agent_evaluate_model()