from util.PPOEnvironment import PPOEnvironment
from util.complex_intersection_utils import process_env, get_env_raw_grid, normalize_env
from util.evaluate_model import parallel_get_reward

model_dir = "models/ppo_grid"

class GridPPOEnvironment(PPOEnvironment):
    
    def __init__(self, model_dir = model_dir):
        output_file = "./outputs/grid/ppo"
        out_csv_file = f"{output_file}/sumo"
        description_args = "PPO Grid"
        super().__init__(output_file, out_csv_file, description_args, model_dir)
    
    def get_env(self, args):
        return process_env(get_env_raw_grid(self.out_csv_file, args))

    def evaluate_model_get_reward(self, model, args):
        return parallel_get_reward(model, args)

if __name__ == "__main__":
    GridPPOEnvironment().save_model()
    GridPPOEnvironment().evaluate_model()