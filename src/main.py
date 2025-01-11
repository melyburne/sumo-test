import simple_intersection 
import grid
import heilbronn 

if __name__ == "__main__":
    """
    Train, save and evaluate DQN and PPO model and log in tensorboard.
    Random agent for baseline.
    In a 2way simple intersection, Heilbronner Oststraße and 4x4 Grid environment.
    """
    simple_intersection.main()
    grid.main()
    heilbronn.main()