from util.PPOEnvironment import PPOEnvironment
from util.complex_intersection_utils import process_env, get_env_raw_heilbronn, normalize_env
from util.evaluate_model import parallel_get_reward

model_dir = "models/ppo_heilbronn"

class HeilbronnPPOEnvironment(PPOEnvironment):
    
    def __init__(self, model_dir = model_dir):
        output_file = "./outputs/heilbronn/ppo"
        out_csv_file = f"{output_file}/sumo"
        description_args = "PPO Heilbronn"
        super().__init__(output_file, out_csv_file, description_args, model_dir)
    
    def get_env(self, args):
        return process_env(normalize_env(get_env_raw_heilbronn(self.out_csv_file, args)))

    def evaluate_model_get_reward(self, model, args):
        return parallel_get_reward(model, args)


if __name__ == "__main__":
    HeilbronnPPOEnvironment().save_model()
    HeilbronnPPOEnvironment().evaluate_model()