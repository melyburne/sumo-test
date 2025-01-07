from util.ParallelRandomEnvironment import ParallelRandomEnvironment
from util.complex_intersection_utils import process_env, get_env_raw_grid, normalize_env

class GridRandomEnvironment(ParallelRandomEnvironment):
    
    def __init__(self):
        output_file = "./outputs/grid/random"
        out_csv_file = f"{output_file}/sumo"
        description_args = "Random Grid"
        super().__init__(output_file, out_csv_file, description_args)
    
    def get_env(self, args):
        return get_env_raw_grid(self.out_csv_file, args)


if __name__ == "__main__":
    GridRandomEnvironment().train_model()
    GridRandomEnvironment().evaluate_model()