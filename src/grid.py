from GridDQNEnvironment import GridDQNEnvironment 
from GridRandomEnvironment import GridRandomEnvironment 
from GridPPOEnvironment import GridPPOEnvironment 

if __name__ == "__main__":
    GridDQNEnvironment().save_model()
    GridPPOEnvironment().save_model()
    GridRandomEnvironment().train_model()

    GridDQNEnvironment().evaluate_model()
    GridPPOEnvironment().evaluate_model()
    GridRandomEnvironment().evaluate_model()