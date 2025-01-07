from util.PPOEnvironment import PPOEnvironment
from util.simple_intersection_utils import get_env

model_dir = "models/ppo_simple-intersection"

class SimpleIntersectionPPOEnvironment(PPOEnvironment):
    
    def __init__(self, model_dir = model_dir):
        output_file = "./outputs/2way-single-intersection/ppo"
        out_csv_file = f"{output_file}/sumo"
        description_args = "PPO Simple-Intersection"
        super().__init__(output_file, out_csv_file, description_args, model_dir)
    
    def get_env(self, args):
        return get_env(self.out_csv_file, args)


if __name__ == "__main__":
    SimpleIntersectionPPOEnvironment().save_model()
    SimpleIntersectionPPOEnvironment().evaluate_model()