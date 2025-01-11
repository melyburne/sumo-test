# Table of Contents

* [heilbronn](#heilbronn)
  * [main](#heilbronn.main)
* [simple\_intersection](#simple_intersection)
  * [main](#simple_intersection.main)
* [HeilbronnDQNEnvironment](#HeilbronnDQNEnvironment)
* [HeilbronnRandomEnvironment](#HeilbronnRandomEnvironment)
* [grid](#grid)
  * [main](#grid.main)
* [GridPPOEnvironment](#GridPPOEnvironment)
* [GridDQNEnvironment](#GridDQNEnvironment)
* [GridRandomEnvironment](#GridRandomEnvironment)
* [main](#main)
* [SimpleIntersectionRandomEnvironment](#SimpleIntersectionRandomEnvironment)
* [HeilbronnPPOEnvironment](#HeilbronnPPOEnvironment)
* [SimpleIntersectionPPOEnvironment](#SimpleIntersectionPPOEnvironment)
* [SimpleIntersectionDQNEnvironment](#SimpleIntersectionDQNEnvironment)
* [util](#util)
* [util.evaluate\_model](#util.evaluate_model)
* [util.RandomEnvironment](#util.RandomEnvironment)
  * [RandomEnvironment](#util.RandomEnvironment.RandomEnvironment)
    * [\_\_init\_\_](#util.RandomEnvironment.RandomEnvironment.__init__)
    * [get\_env](#util.RandomEnvironment.RandomEnvironment.get_env)
    * [get\_random\_args](#util.RandomEnvironment.RandomEnvironment.get_random_args)
    * [run\_model](#util.RandomEnvironment.RandomEnvironment.run_model)
    * [train\_model](#util.RandomEnvironment.RandomEnvironment.train_model)
    * [evaluate\_model](#util.RandomEnvironment.RandomEnvironment.evaluate_model)
* [util.get\_model](#util.get_model)
  * [get\_ppo\_model](#util.get_model.get_ppo_model)
  * [get\_dqn\_model](#util.get_model.get_dqn_model)
* [util.ParallelRandomEnvironment](#util.ParallelRandomEnvironment)
* [util.simple\_intersection\_utils](#util.simple_intersection_utils)
  * [get\_env](#util.simple_intersection_utils.get_env)
* [util.DQNEnvironment](#util.DQNEnvironment)
  * [DQNEnvironment](#util.DQNEnvironment.DQNEnvironment)
    * [\_\_init\_\_](#util.DQNEnvironment.DQNEnvironment.__init__)
    * [get\_env](#util.DQNEnvironment.DQNEnvironment.get_env)
    * [get\_dqn\_args](#util.DQNEnvironment.DQNEnvironment.get_dqn_args)
    * [train\_model](#util.DQNEnvironment.DQNEnvironment.train_model)
    * [save\_model](#util.DQNEnvironment.DQNEnvironment.save_model)
    * [evaluate\_model](#util.DQNEnvironment.DQNEnvironment.evaluate_model)
* [util.file\_utils](#util.file_utils)
* [util.PPOEnvironment](#util.PPOEnvironment)
  * [PPOEnvironment](#util.PPOEnvironment.PPOEnvironment)
    * [\_\_init\_\_](#util.PPOEnvironment.PPOEnvironment.__init__)
    * [get\_env](#util.PPOEnvironment.PPOEnvironment.get_env)
    * [get\_ppo\_args](#util.PPOEnvironment.PPOEnvironment.get_ppo_args)
    * [train\_model](#util.PPOEnvironment.PPOEnvironment.train_model)
    * [save\_model](#util.PPOEnvironment.PPOEnvironment.save_model)
    * [evaluate\_model](#util.PPOEnvironment.PPOEnvironment.evaluate_model)
* [util.RandomActionModel](#util.RandomActionModel)
  * [RandomActionModel](#util.RandomActionModel.RandomActionModel)
    * [predict](#util.RandomActionModel.RandomActionModel.predict)
    * [train](#util.RandomActionModel.RandomActionModel.train)
    * [save](#util.RandomActionModel.RandomActionModel.save)
* [util.random\_agent](#util.random_agent)
  * [log\_tensorboard\_train](#util.random_agent.log_tensorboard_train)
* [util.parse\_args](#util.parse_args)
  * [parse\_args](#util.parse_args.parse_args)
  * [parse\_args\_model](#util.parse_args.parse_args_model)
  * [parse\_args\_evaluate](#util.parse_args.parse_args_evaluate)
  * [parse\_args\_ppo](#util.parse_args.parse_args_ppo)
  * [parse\_args\_dqn](#util.parse_args.parse_args_dqn)
* [util.complex\_intersection\_utils](#util.complex_intersection_utils)
  * [process\_env](#util.complex_intersection_utils.process_env)
  * [normalize\_env](#util.complex_intersection_utils.normalize_env)
  * [get\_env\_raw](#util.complex_intersection_utils.get_env_raw)
  * [get\_env\_raw\_heilbronn](#util.complex_intersection_utils.get_env_raw_heilbronn)
  * [get\_env\_raw\_grid](#util.complex_intersection_utils.get_env_raw_grid)

<a id="heilbronn"></a>

# heilbronn

<a id="heilbronn.main"></a>

#### main

```python
def main()
```

Train, save and evaluate DQN and PPO model and log in tensorboard.
Random agent for baseline.
In a Heilbronner Oststraße environment.

<a id="simple_intersection"></a>

# simple\_intersection

<a id="simple_intersection.main"></a>

#### main

```python
def main()
```

Train, save and evaluate DQN and PPO model and log in tensorboard.
Random agent for baseline.
In a 2way simple intersection environment.

<a id="HeilbronnDQNEnvironment"></a>

# HeilbronnDQNEnvironment

See [util.DQNEnvironment](#util.DQNEnvironment).

<a id="HeilbronnRandomEnvironment"></a>

# HeilbronnRandomEnvironment

See [util.RandomEnvironment](#util.RandomEnvironment).

<a id="grid"></a>

# grid

<a id="grid.main"></a>

#### main

```python
def main()
```

Train, save and evaluate DQN and PPO model and log in tensorboard.
Random agent for baseline.
In a 4x4 Grid environment.

<a id="GridPPOEnvironment"></a>

# GridPPOEnvironment

See [util.PPOEnvironment](#util.PPOEnvironment).

<a id="GridDQNEnvironment"></a>

# GridDQNEnvironment

See [util.DQNEnvironment](#util.DQNEnvironment).

<a id="GridRandomEnvironment"></a>

# GridRandomEnvironment

See [util.RandomEnvironment](#util.RandomEnvironment).

<a id="main"></a>

# main

<a id="SimpleIntersectionRandomEnvironment"></a>

# SimpleIntersectionRandomEnvironment

See [util.RandomEnvironment](#util.RandomEnvironment).

<a id="HeilbronnPPOEnvironment"></a>

# HeilbronnPPOEnvironment

See [util.PPOEnvironment](#util.PPOEnvironment).

<a id="SimpleIntersectionPPOEnvironment"></a>

# SimpleIntersectionPPOEnvironment

See [util.PPOEnvironment](#util.PPOEnvironment).

<a id="SimpleIntersectionDQNEnvironment"></a>

# SimpleIntersectionDQNEnvironment

See [util.DQNEnvironment](#util.DQNEnvironment).

<a id="util"></a>

# util

<a id="util.evaluate_model"></a>

# util.evaluate\_model

<a id="util.RandomEnvironment"></a>

# util.RandomEnvironment

<a id="util.RandomEnvironment.RandomEnvironment"></a>

## RandomEnvironment Objects

```python
class RandomEnvironment(ABC)
```

<a id="util.RandomEnvironment.RandomEnvironment.__init__"></a>

#### \_\_init\_\_

```python
def __init__(output_file, out_csv_file, description_args)
```

Initialize the Sumo environment and let the agent perform a random action with it.

**Arguments**:

- `output_file`: Path where output files of tensorboard will be saved
- `out_csv_file`: Path where the output CSV file of SUMO will be saved
- `description_args`: Description of the ArgumentParser

<a id="util.RandomEnvironment.RandomEnvironment.get_env"></a>

#### get\_env

```python
@abstractmethod
def get_env(args)
```

Return the environment for the given arguments.

The class uses the returned environment to execute the agent.

**Arguments**:

- `args`: ArgumentParser instance with the required arguments.

<a id="util.RandomEnvironment.RandomEnvironment.get_random_args"></a>

#### get\_random\_args

```python
def get_random_args()
```

Returns the ArgumentParser instance with the arguments required for the agent to work.

<a id="util.RandomEnvironment.RandomEnvironment.run_model"></a>

#### run\_model

```python
@abstractmethod
def run_model(env, args, total_timesteps, seconds)
```

Defines how to agent interact with the environment.

**Arguments**:

- `env`: SumoEnvironment for the agent to interact with.
- `args`: ArgumentParser instance with the required arguments.
- `total_timesteps`: The total number of samples (env steps) to train on
- `seconds`: Number of simulated seconds on SUMO. The duration in seconds of the simulation.

<a id="util.RandomEnvironment.RandomEnvironment.train_model"></a>

#### train\_model

```python
def train_model()
```

Agent interact with the environment and the results will be logged as output. 
This function is used as a baseline to compare models while training.

<a id="util.RandomEnvironment.RandomEnvironment.evaluate_model"></a>

#### evaluate\_model

```python
def evaluate_model(ep_length=200)
```

Agent interact with the environment and the results will be logged as output. 
This function is used as a baseline to compare models while evaluation.

<a id="util.get_model"></a>

# util.get\_model

<a id="util.get_model.get_ppo_model"></a>

#### get\_ppo\_model

```python
def get_ppo_model(env, args, output_file)
```

Returns a PPO model.

**Arguments**:

- `env`: Gym environment for the agent to interact with.
- `args`: ArgumentParser instance with the required arguments.
- `output_file`: Path where output files of tensorboard will be saved

<a id="util.get_model.get_dqn_model"></a>

#### get\_dqn\_model

```python
def get_dqn_model(env, args, output_file)
```

Returns a DQN model.

**Arguments**:

- `env`: Gym environment for the agent to interact with.
- `args`: ArgumentParser instance with the required arguments.
- `output_file`: Path where output files of tensorboard will be saved

<a id="util.ParallelRandomEnvironment"></a>

# util.ParallelRandomEnvironment

See [util.RandomEnvironment](#util.RandomEnvironment).

<a id="util.simple_intersection_utils"></a>

# util.simple\_intersection\_utils

<a id="util.simple_intersection_utils.get_env"></a>

#### get\_env

```python
def get_env(out_csv_file, args)
```

Return the environment of a simple intersection for the given arguments.

**Arguments**:

- `args`: ArgumentParser instance with the required arguments.

<a id="util.DQNEnvironment"></a>

# util.DQNEnvironment

<a id="util.DQNEnvironment.DQNEnvironment"></a>

## DQNEnvironment Objects

```python
class DQNEnvironment(ABC)
```

<a id="util.DQNEnvironment.DQNEnvironment.__init__"></a>

#### \_\_init\_\_

```python
def __init__(output_file, out_csv_file, description_args, model_dir)
```

Initialize the Sumo environment and let the DQN model train, save and evaluate.

**Arguments**:

- `output_file`: Path where output files of tensorboard will be saved
- `out_csv_file`: Path where the output CSV file of SUMO will be saved
- `description_args`: Description of the ArgumentParser
- `model_dir`: Directory to save the model

<a id="util.DQNEnvironment.DQNEnvironment.get_env"></a>

#### get\_env

```python
@abstractmethod
def get_env(args)
```

Return the environment for the given arguments.

The class uses the returned environment to execute the agent.

**Arguments**:

- `args`: ArgumentParser instance with the required arguments.

<a id="util.DQNEnvironment.DQNEnvironment.get_dqn_args"></a>

#### get\_dqn\_args

```python
def get_dqn_args()
```

Returns the ArgumentParser instance with the arguments required for the agent to work.

<a id="util.DQNEnvironment.DQNEnvironment.train_model"></a>

#### train\_model

```python
def train_model(env, args)
```

Train the model.

**Arguments**:

- `env`: Gym environment for the agent to interact with.
- `args`: ArgumentParser instance with the required arguments.

<a id="util.DQNEnvironment.DQNEnvironment.save_model"></a>

#### save\_model

```python
def save_model()
```

Train and save a model and log in tensorboard.
Environment defined in method get_env.
The constructor parameter model_dir determines the storage location of the model.

<a id="util.DQNEnvironment.DQNEnvironment.evaluate_model"></a>

#### evaluate\_model

```python
def evaluate_model()
```

Evaluate a model and log in tensorboard.
Environment defined in method get_env.
The model used is defined in the constructor parameter model_dir.

<a id="util.file_utils"></a>

# util.file\_utils

<a id="util.PPOEnvironment"></a>

# util.PPOEnvironment

<a id="util.PPOEnvironment.PPOEnvironment"></a>

## PPOEnvironment Objects

```python
class PPOEnvironment(ABC)
```

<a id="util.PPOEnvironment.PPOEnvironment.__init__"></a>

#### \_\_init\_\_

```python
def __init__(output_file, out_csv_file, description_args, model_dir)
```

Initialize the Sumo environment and let the PPO model train, save and evaluate.

**Arguments**:

- `output_file`: Path where output files of tensorboard will be saved
- `out_csv_file`: Path where the output CSV file of SUMO will be saved
- `description_args`: Description of the ArgumentParser
- `model_dir`: Directory to save the model

<a id="util.PPOEnvironment.PPOEnvironment.get_env"></a>

#### get\_env

```python
@abstractmethod
def get_env(args)
```

Return the environment for the given arguments.

The class uses the returned environment to execute the agent.

**Arguments**:

- `args`: ArgumentParser instance with the required arguments.

<a id="util.PPOEnvironment.PPOEnvironment.get_ppo_args"></a>

#### get\_ppo\_args

```python
def get_ppo_args()
```

Returns the ArgumentParser instance with the arguments required for the agent to work.

<a id="util.PPOEnvironment.PPOEnvironment.train_model"></a>

#### train\_model

```python
def train_model(env, args)
```

Train the model.

**Arguments**:

- `env`: Gym environment for the agent to interact with.
- `args`: ArgumentParser instance with the required arguments.

<a id="util.PPOEnvironment.PPOEnvironment.save_model"></a>

#### save\_model

```python
def save_model()
```

Train and save a model and log in tensorboard.
Environment defined in method get_env.
The constructor parameter model_dir determines the storage location of the model.

<a id="util.PPOEnvironment.PPOEnvironment.evaluate_model"></a>

#### evaluate\_model

```python
def evaluate_model()
```

Evaluate a model and log in tensorboard.
Environment defined in method get_env.
The model used is defined in the constructor parameter model_dir.

<a id="util.RandomActionModel"></a>

# util.RandomActionModel

<a id="util.RandomActionModel.RandomActionModel"></a>

## RandomActionModel Objects

```python
class RandomActionModel()
```

A simple model that selects random actions.

<a id="util.RandomActionModel.RandomActionModel.predict"></a>

#### predict

```python
def predict(_)
```

Returns a random action.

<a id="util.RandomActionModel.RandomActionModel.train"></a>

#### train

```python
def train(*args, **kwargs)
```

No training required for random actions.

<a id="util.RandomActionModel.RandomActionModel.save"></a>

#### save

```python
def save(path)
```

Save the random model as metadata.

<a id="util.random_agent"></a>

# util.random\_agent

<a id="util.random_agent.log_tensorboard_train"></a>

#### log\_tensorboard\_train

```python
def log_tensorboard_train(output_file, episode_rewards, seconds)
```

Log the rewards in tensorboard.

**Arguments**:

- `output_file`: Path where output files of tensorboard will be saved.
- `seconds`: Number of simulated seconds on SUMO. The duration in seconds of the simulation.

<a id="util.parse_args"></a>

# util.parse\_args

<a id="util.parse_args.parse_args"></a>

#### parse\_args

```python
def parse_args(description)
```

Returns the ArgumentParser instance with the arguments required for an agent and SUMO to work.

**Arguments**:

- `description`: Description of the ArgumentParser

<a id="util.parse_args.parse_args_model"></a>

#### parse\_args\_model

```python
def parse_args_model(prs)
```

Returns the ArgumentParser instance with additional arguments required for an model to train.

**Arguments**:

- `prs`: ArgumentParser, which requires the additional arguments

<a id="util.parse_args.parse_args_evaluate"></a>

#### parse\_args\_evaluate

```python
def parse_args_evaluate(prs)
```

Returns the ArgumentParser instance with additional arguments required for an model to evaluate.

**Arguments**:

- `prs`: ArgumentParser, which requires the additional arguments

<a id="util.parse_args.parse_args_ppo"></a>

#### parse\_args\_ppo

```python
def parse_args_ppo(prs)
```

Returns the ArgumentParser instance with additional arguments required for a PPO model.

**Arguments**:

- `prs`: ArgumentParser, which requires the additional arguments

<a id="util.parse_args.parse_args_dqn"></a>

#### parse\_args\_dqn

```python
def parse_args_dqn(prs)
```

Returns the ArgumentParser instance with additional arguments required for a DQN model.

**Arguments**:

- `prs`: ArgumentParser, which requires the additional arguments

<a id="util.complex_intersection_utils"></a>

# util.complex\_intersection\_utils

<a id="util.complex_intersection_utils.process_env"></a>

#### process\_env

```python
def process_env(env)
```

Convert PettingZooAPI to Gym Environment.

**Arguments**:

- `env`: PettingoZoo environment.

<a id="util.complex_intersection_utils.normalize_env"></a>

#### normalize\_env

```python
def normalize_env(env)
```

Ensure compatibility in multi-agent environments by padding the action and observation spaces so all agents have uniform spaces.

**Arguments**:

- `env`: PettingoZoo environment.

<a id="util.complex_intersection_utils.get_env_raw"></a>

#### get\_env\_raw

```python
def get_env_raw(out_csv_file, args, net_file, route_file)
```

Returns the environment for the specified arguments and files.

**Arguments**:

- `out_csv_file`: Path where the output CSV file of SUMO will be saved
- `args`: ArgumentParser instance with the required arguments.
- `net_file`: Path to SUMO .net.xml file
- `route_file`: Path to SUMO .rou.xml file

<a id="util.complex_intersection_utils.get_env_raw_heilbronn"></a>

#### get\_env\_raw\_heilbronn

```python
def get_env_raw_heilbronn(out_csv_file, args)
```

Return the environment of a Heilbronner Oststraße for the given arguments.

**Arguments**:

- `out_csv_file`: Path where the output CSV file of SUMO will be saved
- `args`: ArgumentParser instance with the required arguments.

<a id="util.complex_intersection_utils.get_env_raw_grid"></a>

#### get\_env\_raw\_grid

```python
def get_env_raw_grid(out_csv_file, args)
```

Return the environment of a 4x4 grid for the given arguments.

**Arguments**:

- `out_csv_file`: Path where the output CSV file of SUMO will be saved
- `args`: ArgumentParser instance with the required arguments.

