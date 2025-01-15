# General Arguments in SUMO

| Argument Name | Type | Default Value | Description                                                           |
| ------------- | ---- | ------------- | --------------------------------------------------------------------- |
| -mingreen     | int  | 10            | Minimum green time.                                                   |
| -maxgreen     | int  | 30            | Maximum green time.                                                   |
| -gui          | bool | False         | Run with visualization on SUMO.                                       |
| -s            | int  | 1000          | Number of simulation seconds in Sumo. That's how long an episode lasts. |

# General Arguments for Model Training (including Baseline Random Agent)

| Argument Name | Type | Default Value | Description                             |
| ------------- | ---- | ------------- | --------------------------------------- |
| -tt           | int  | 100000        | Total timesteps for the model learning. |

# General Arguments for Evaluating Model

| Argument Name | Type | Default Value | Description                              |
| ------------- | ---- | ------------- | ---------------------------------------- |
| -nee          | int  | 10            | Number of episodes to evaluate the agent |

# Arguments for PPO and DQN Model

| Argument Name | Type  | Default Value | Description                                           |
| ------------- | ----- | ------------- | ----------------------------------------------------- |
| -lr           | float | 0.0003        | Learning rate of the model.                           |
| -g            | float | 0.99          | Discount factor of the model.                         |
| -bs           | int   | 64            | Minibatch size for each gradient update of the model. |

# Arguments for PPO Model

| Argument Name | Type  | Default Value | Description                                                                                    |
| ------------- | ----- | ------------- | ---------------------------------------------------------------------------------------------- |
| -ns           | int   | 256           | The number of steps to run for each environment per update in the PPO model.                   |
| -gl           | float | 0.95          | Factor for trade-off of bias vs variance for Generalized Advantage Estimator in the PPO model. |
| -cr           | float | 0.95          | Clipping parameter in the PPO model.                                                           |

# Arguments for DQN Model

| Argument Name | Type  | Default Value | Description                                                |
| ------------- | ----- | ------------- | ---------------------------------------------------------- |
| -ls           | int   | 0             | Number of steps before learning starts in the DQN model.   |
| -tf           | int   | 1             | Training frequency of the DQN model.                       |
| -tui          | int   | 500           | Interval for updating the target network in the DQN model. |
| -eie          | float | 0.05          | Initial exploration epsilon for the DQN model.             |
| -efe          | float | 0.01          | Final exploration epsilon for the DQN model.               |
