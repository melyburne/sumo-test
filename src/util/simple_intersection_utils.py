from sumo_rl import SumoEnvironment
from .file_utils import get_file

def get_env(out_csv_file, args):
    """
        Return the environment of a simple intersection for the given arguments.

        :param args: ArgumentParser instance with the required arguments.
    """
    net_file = get_file('two-way-single-intersection.net.xml')
    route_file = get_file('two-way-single-intersection-gen.rou.xml')
    return SumoEnvironment(
        net_file=net_file,
        route_file=route_file,
        out_csv_name=out_csv_file,
        use_gui=args.gui,
        num_seconds=args.seconds,
        min_green=args.min_green,
        max_green=args.max_green,
        yellow_time=4,
        single_agent=True,
        reward_fn="diff-waiting-time"
    )

