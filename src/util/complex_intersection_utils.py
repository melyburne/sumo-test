from .file_utils import get_file

from stable_baselines3.common.vec_env import VecMonitor

import supersuit as ss
import sumo_rl

import gym

def get_env(out_csv_file, args):
    env = get_env_raw(out_csv_file, args)
    env.unwrapped.render_mode = 'human'

    env = ss.multiagent_wrappers.pad_action_space_v0(env)
    env = ss.multiagent_wrappers.pad_observations_v0(env)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, 1, num_cpus=1, base_class="stable_baselines3")
    env = VecMonitor(env)

    return env

def get_env_raw(out_csv_file, args):
    # net_file = get_file('4x4.net.xml')
    # route_file = get_file('4x4c1c2c1c2.rou.xml')

    net_file = get_file('heilbronn.net.xml.gz')
    route_file = get_file('heilbronn.passenger.trips.xml')
    
    env = sumo_rl.parallel_env(
        net_file=net_file,
        route_file=route_file,
        out_csv_name=out_csv_file,
        use_gui=args.gui,
        num_seconds=args.seconds,
        min_green=args.min_green,
        max_green=args.max_green,
        yellow_time=4,
        reward_fn="diff-waiting-time",
    )

    return env

