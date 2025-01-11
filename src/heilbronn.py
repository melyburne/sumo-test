from HeilbronnDQNEnvironment import HeilbronnDQNEnvironment 
from HeilbronnRandomEnvironment import HeilbronnRandomEnvironment 
from HeilbronnPPOEnvironment import HeilbronnPPOEnvironment 

def main():
    """
    Train, save and evaluate DQN and PPO model and log in tensorboard.
    Random agent for baseline.
    In a Heilbronner Oststraße environment.
    """
    HeilbronnDQNEnvironment().save_model()
    HeilbronnPPOEnvironment().save_model()
    HeilbronnRandomEnvironment().train_model()

    HeilbronnDQNEnvironment().evaluate_model()
    HeilbronnPPOEnvironment().evaluate_model()
    HeilbronnRandomEnvironment().evaluate_model()

if __name__ == "__main__":
    main()