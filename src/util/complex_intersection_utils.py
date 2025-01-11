from .file_utils import get_file

from stable_baselines3.common.vec_env import VecMonitor

import supersuit as ss
import sumo_rl

import gym

def process_env(env):
    """
    Convert PettingZooAPI to Gym Environment.

    :param env: PettingoZoo environment.
    """
    env.unwrapped.render_mode = 'human'
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, 1, num_cpus=1, base_class="stable_baselines3")
    env = VecMonitor(env)
    return env


def normalize_env(env):
    """
    Ensure compatibility in multi-agent environments by padding the action and observation spaces so all agents have uniform spaces.

    :param env: PettingoZoo environment.
    """
    env = ss.multiagent_wrappers.pad_action_space_v0(env)
    env = ss.multiagent_wrappers.pad_observations_v0(env)
    return env


def get_env_raw(out_csv_file, args, net_file, route_file):
    """
        Returns the environment for the specified arguments and files.

        :param out_csv_file: Path where the output CSV file of SUMO will be saved
        :param args: ArgumentParser instance with the required arguments.
        :param net_file: Path to SUMO .net.xml file
        :param route_file: Path to SUMO .rou.xml file
    """

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


def get_env_raw_heilbronn(out_csv_file, args):
    """
        Return the environment of a Heilbronner Oststraße for the given arguments.

        :param out_csv_file: Path where the output CSV file of SUMO will be saved
        :param args: ArgumentParser instance with the required arguments.
    """
    net_file = get_file('heilbronn.net.xml.gz')
    route_file = get_file('heilbronn.passenger.trips.xml')

    return get_env_raw(out_csv_file, args, net_file, route_file)


def get_env_raw_grid(out_csv_file, args):
    """
        Return the environment of a 4x4 grid for the given arguments.

        :param out_csv_file: Path where the output CSV file of SUMO will be saved
        :param args: ArgumentParser instance with the required arguments.
    """
    net_file = get_file('4x4.net.xml')
    route_file = get_file('4x4c1c2c1c2.rou.xml')

    return get_env_raw(out_csv_file, args, net_file, route_file)

