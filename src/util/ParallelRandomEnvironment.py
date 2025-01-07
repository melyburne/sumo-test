from .RandomEnvironment import RandomEnvironment
from .RandomActionModel import RandomActionModel

class ParallelRandomEnvironment(RandomEnvironment):

    def __init__(self, output_file, out_csv_file, description_args):
        """
        Initialize the environment with specific parameters.
        
        :param output_file: Path where output files will be saved
        :param out_csv_file: Path to the CSV file used for the environment
        :param description_args: Description of the environment
        :param model_dir: Directory to save the model
        """
        self.output_file = output_file
        self.out_csv_file = out_csv_file
        self.description_args = description_args
        super().__init__(output_file, out_csv_file, description_args)

    def run_model(self, env, args, total_timesteps, seconds):
        # Initialize the random action model
        model = RandomActionModel(env)

        obs = env.reset()
        simulationDone = False
        total_reward = 0
        step_count = 0

        current_episode = 0
        episode_rewards = []

        while not simulationDone:
            # Use the random model to decide an action
            actions = {agent: env.action_space(agent).sample() for agent in env.agents}

            # Apply the action and get the next observation and reward
            obs, reward, done, truncated, info = env.step(actions)

            # Accumulate reward and count steps
            total_reward += sum(reward.values()) 
            step_count += 1

            # Check if the episode is done (based on `seconds` or `done` condition)
            if (step_count % seconds) == 0:
                # Add the total reward for this episode to the list
                episode_rewards.append(total_reward)

                # Reset for the next episode
                current_episode += 1
                obs = env.reset()
                total_reward = 0  # Reset total reward for the new episode

            # Stop the loop if the total timesteps have been reached
            if step_count == total_timesteps:
                simulationDone = True

        env.close()

        return episode_rewards  # Return the list of rewards
