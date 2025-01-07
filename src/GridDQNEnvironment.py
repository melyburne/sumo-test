from util.DQNEnvironment import DQNEnvironment
from util.complex_intersection_utils import process_env, get_env_raw_grid, normalize_env

model_dir = "models/dqn_grid"

class GridDQNEnvironment(DQNEnvironment):
    
    def __init__(self, model_dir = model_dir):
        output_file = "./outputs/grid/dqn"
        out_csv_file = f"{output_file}/sumo"
        description_args = "DQN Grid"
        super().__init__(output_file, out_csv_file, description_args, model_dir)
    
    def get_env(self, args):
        return process_env(get_env_raw_grid(self.out_csv_file, args))


if __name__ == "__main__":
    GridDQNEnvironment().save_model()
    GridDQNEnvironment().evaluate_model()