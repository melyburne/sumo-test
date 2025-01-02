class RandomActionModel:
    """A simple model that selects random actions."""
    def __init__(self, env):
        self.env = env

    def predict(self, _):
        """Returns a random action."""
        action = self.env.action_space.sample()
        return action, None  # Return action and state (None for random actions)

    def train(self, *args, **kwargs):
        """No training required for random actions."""
        pass

    def save(self, path):
        """Save the random model as metadata."""
        with open(path, "w") as f:
            f.write("Random action model - no learnable parameters.")