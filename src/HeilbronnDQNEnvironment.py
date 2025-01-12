from util.DQNEnvironment import DQNEnvironment
from util.complex_intersection_utils import process_env, get_env_raw_heilbronn, normalize_env
from util.evaluate_model import parallel_get_reward

model_dir = "models/dqn_heilbronn"

class HeilbronnDQNEnvironment(DQNEnvironment):
    
    def __init__(self, model_dir = model_dir):
        output_file = "./outputs/heilbronn/dqn"
        out_csv_file = f"{output_file}/sumo"
        description_args = "DQN Heilbronn"
        super().__init__(output_file, out_csv_file, description_args, model_dir)
    
    def get_env(self, args):
        return process_env(normalize_env(get_env_raw_heilbronn(self.out_csv_file, args)))

    def evaluate_model_get_reward(self, model, args):
        return parallel_get_reward(model, args)


if __name__ == "__main__":
    HeilbronnDQNEnvironment().save_model()
    HeilbronnDQNEnvironment().evaluate_model()