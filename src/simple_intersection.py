from SimpleIntersectionDQNEnvironment import SimpleIntersectionDQNEnvironment 
from SimpleIntersectionRandomEnvironment import SimpleIntersectionRandomEnvironment 
from SimpleIntersectionPPOEnvironment import SimpleIntersectionPPOEnvironment 

def main():
    """
    Train, save and evaluate DQN and PPO model and log in tensorboard.
    Random agent for baseline.
    In a 2way simple intersection environment.
    """
    SimpleIntersectionDQNEnvironment().save_model()
    SimpleIntersectionPPOEnvironment().save_model()
    SimpleIntersectionRandomEnvironment().train_model()

    SimpleIntersectionDQNEnvironment().evaluate_model()
    SimpleIntersectionPPOEnvironment().evaluate_model()
    SimpleIntersectionRandomEnvironment().evaluate_model()

if __name__ == "__main__":
    main()
    