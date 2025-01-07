from util.ParallelRandomEnvironment import ParallelRandomEnvironment
from util.complex_intersection_utils import get_env_raw_heilbronn

class HeilbronnRandomEnvironment(ParallelRandomEnvironment):
    
    def __init__(self):
        output_file = "./outputs/heilbronn/random"
        out_csv_file = f"{output_file}/sumo"
        description_args = "Random Heilbronn"
        super().__init__(output_file, out_csv_file, description_args)
    
    def get_env(self, args):
        return get_env_raw_heilbronn(self.out_csv_file, args)


if __name__ == "__main__":
    HeilbronnRandomEnvironment().train_model()
    HeilbronnRandomEnvironment().evaluate_model()