This project uses SUMO and sumo-rl to train and evaluate a PPO and DQN model that controls the traffic lights to reduce the differences in waiting times per lane in a traffic situation. The baseline is an agent that performs random actions. The project uses three different maps: a map of Heilbronn's Oststraße, a map of a single-lane 4x4 grid and a map of a two-lane single intersection.

# Requirements

The requirements of the project can be found in requirements.txt. The project was tested using the shell.nix which uses Python 3.12.7 and the requirements defined in tested_requirements.txt.

## SUMO

SUMO must also be installed for the project. Further information can be found in [the official documentation](https://sumo.dlr.de/docs/Installing/index.html). In order for the project to find sumo, SUMO_HOME must be set to the folder in which sumo is located.
Under Linux, cmd would be as follows 
```bash
export SUMO_HOME="usr/share/sumo”
``` 

# Python Scripts to execute

## Main Scripts

| Python script              | What it does                                                                                                                                                                         |
| -------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| src/main.py                | Train and evaluate a PPO and DQN model with the random agent as baseline; using the one-lane 4x4 grid, the two-lane single intersection and the Heilbronner Oststraße as environment |
| src/grid.py                | Train and evaluate a PPO and DQN model with the random agent as baseline; using the single-lane 4x4 grid as environment                                                              |
| src/simple_intersection.py | Train and evaluate a PPO and DQN model with the random agent as baseline; using a two-lane single-lane intersection as environment                                                   |
| src/heilbronn.py           | Train and evaluate a PPO and DQN model with the random agent as baseline; using Heilbronner Oststraße as environment                                                                 |

## Other Scripts

The scripts execute following model/agent in the following environment:

|                                   | DQN Model                               | PPO Model                               | Random agent as baseline                   |
| --------------------------------- | --------------------------------------- | --------------------------------------- | ------------------------------------------ |
| Heilbronner Oststraße             | src/HeilbronnDQNEnvironment.py          | src/HeilbronnPPOEnvironment.py          | src/HeilbronnRandomEnvironment.py          |
| Two-lane Single-lane Intersection | src/SimpleIntersectionDQNEnvironment.py | src/SimpleIntersectionPPOEnvironment.py | src/SimpleIntersectionRandomEnvironment.py |
| Single-lane 4x4 Grid              | src/GridDQNEnvironment.py               | src/GridPPOEnvironment.py               | src/GridRandomEnvironment.py               |

## Miscellaneous

- The documentation of the scripts can be found in docs.md. 
- Which args can be used with the scripts can be found in docs_args.md.

# Output

Normally, the output files are located in the output folder. A sample output can be found in the outputs_example folder, which contains the SUMO environment outputs in csv format and the tensorboard output files, and in the model_example folder, which contains the saved PPO and DQN models for each environment.
To see tensorbaord, execute cmd 
``bash
tensorboard --logdir=”./outputs_example”
```

