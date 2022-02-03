# Maze-RL L2RPN - ICAPS 2021 Submission

![Example for trained agent in action](https://drive.google.com/uc?export=download&id=1J9ysSmqjKXbkpxE8eb2U_EzvbNspIYEH)

The ["Learning to run a power network" (L2RPN)](https://l2rpn.chalearn.org/) challenge is a series of competitions 
organized by RTE, the French Transmition System Operator with the aim to test the potential of 
reinforcement learning (RL) to control electrical power transmission. The challenge is motivated by the fact that 
existing methods are not adequate for real-time network operations on short temporal horizons in a reasonable compute 
time. Also, power networks are facing a steadily growing share of renewable energy, requiring faster responses. This 
raises the need for highly robust and adaptive power grid controllers.

In 2021, [the L2RPN competition](https://competitions.codalab.org/competitions/33121) was held at the  International 
Conference on Automated Planning and Scheduling ([ICAPS](https://icaps21.icaps-conference.org/home/)). This repository 
contains the submission of [enliteAI's](https://www.enlite.ai/) RL-team which ranked 3rd on the official leaderboard 
(maze-rl). The code in this repository builds on our [RL framework Maze](https://github.com/enlite-ai/maze).

For a Maze-RL implementation of the winning solution from an early L2RPN challenge please refer to the 
[gitbub repository](https://github.com/enlite-ai/maze_smaac) or for a more extensive wrap up you can also check out our
[accompanying blog post](https://enliteai.medium.com/reinforcing-power-grids-a-baseline-implementation-for-l2rpn-830401fd2e62).

### Overview
* [Installation](#section-installation)
* [Run Rollout of the Agent](#section-rollout)
* [Test Submission](#section-test-submission)
* [Directory Structure](#section-directory-structure)
* [About the RL Framework Maze](#section-maze)

<a name="section-installation"></a> 
## Installation

Install all dependencies:
```shell
conda env create -f environment.yml
conda activate maze_l2rpn_icaps_2021
```

Install [lightsim2grid](https://github.com/BDonnot/lightsim2grid), a fast backend for [Grid2Op](https://github.com/rte-france/Grid2Op):
```shell
chmod +x install_lightsim2grid.sh
./install_lightsim2grid.sh
```
<a name="section-rollout"></a>
## Run Rollout of the Agent

If everything is installed correctly, this code snipped performs a rollout of our final agent on the local grid2op 
environment. (Please note that some data will be downloaded if this is the first time the 'l2rpn_icaps_2021_small' 
env is being built.)

```python
import grid2op
from lightsim2grid import LightSimBackend
from submission_icaps_2021.maze_agent.submission import make_agent

# init environment and backend
env = grid2op.make("l2rpn_icaps_2021_small", difficulty="competition", backend=LightSimBackend())
# initialize agent
agent = make_agent(grid2op_env=env,
                   this_directory_path='<path/to/this/directory>')

# reset env and run interaction loop
obs = env.reset()
step_count, rew, done, info = 0, 0.0, False, {}
while not done:
    action = agent.act(observation=obs, reward=rew, done=False)
    obs, rew, done, info = env.step(action)
    step_count += 1
print(f'survived {step_count} steps')
```
If pygraphviz was installed as well, building the agent also compiles a graphical representation of the policy network:

![Graphical representation of our policy network](https://drive.google.com/uc?export=download&id=16YUSOSIygiJcWwPcYxZy8m8IsdWvE5ia)

<a name="section-test-submission"></a>
## Test Submission

To build an actual submission run::

```shell
python submission_icaps_2021/compose_submission.py
```

This command will create a temporary folder at the default location _/tmp/maze_l2rpn_submission_ where all necessary 
files and packages will be copied. Once the submission directory is created, the submission is checked by running the 
_check_submission_ method provided by the organizers of the challenge. This method creates a zip file for online
submission and tests the agent with local development data.

Your console output should then look like this:
```shell
Your scores are :
(remember these score are not at all an indication of what will be used in codalab, as the data it is tested on are really different):"

                 0            1
0            score    47.828603
1         duration   315.333469
2  total_operation    47.278956
3  total_attention    49.111111
```
<a name="section-directory-structure"></a>
## Directory Structure

    .
    ├── experiment_data
    │   ├── .hydra
    │   │   └── config.yaml             # The experiment config file
    │   ├── obs_norm_statistics.pkl     # The estimated observation normalization statistics 
    │   ├── spaces_config.pkl           # The obstavtion/action spaces configuration
    │   └── state_dict.pt               # The trained weights of the policy
    │
    ├── images                          # Images displayed in the readme file
    │
    ├── maze_l2rpn                      # The maze_l2rpn code base
    │   ├── agents                      # The random agent
    │   ├── env                         # The enviroment components
    │   ├── model                       # The torch model used
    │   ├── reward                      # The reward aggregator
    │   ├── space_interfaces            # Observation and action spaces interfacing the maze env with the grid2op env
    │   ├── wrappers                    # The wrappers used for training and rollout
    │   ├── __init__.py                 
    │   └── utils.py                    # The utilities file
    │
    ├── submission_icaps_2021           # The submission system for the icaps 2021 competition parts of which have been updated
    │   ├── input_data_local            # Data provided by the organizers for creating and testing the submission
    │   ├── maze_agents                 # Our agent used for the competition
    │   ├── res_alreat                  # Data provided by the organizers for creating and testing the submission
    │   ├── utils                       # Utilities provided by the organizers for creating and testing the submission
    │   ├── check_your_submission.py    # Utility provided by the organizers for creating and testing the submission
    │   ├── get_diff_import.py          # Utility for finding all additional packages need for the submission to run on the server.
    │   ├── get_info_import.py          # Utility provided by the organizers for creating and testing the submission
    │   └── requirements.txt            # A list of packages installel on the icaps_2021 test server
    │
    ├── .gitignore                      
    ├── environment.yml                 # The conda envornment specification for running the code in this repo
    ├── install_lightsim2grid.sg        # A small scipt for installing the correct lightsim2grid version from source
    ├── LICENCE.txt                     # The Enlite.AI reserach only licence applicaple to all code in this repository
    ├── README.md                       # This file
    └── setup.py                        # pip setup file for maze_l2rpn

<a name="section-maze"></a>
## About the RL Framework Maze

![Banner](https://github.com/enlite-ai/maze/raw/main/docs/source/logos/main_logo.png)

[Maze](https://github.com/enlite-ai/maze) is an application-oriented deep reinforcement learning (RL) framework, addressing real-world decision problems.
Our vision is to cover the complete development life-cycle of RL applications, ranging from simulation engineering to agent development, training and deployment.
  
If you encounter a bug, miss a feature or have a question that the [documentation](https://maze-rl.readthedocs.io/) doesn't answer: We are happy to assist you! Report an [issue](https://github.com/enlite-ai/maze/issues) or start a discussion on [GitHub](https://github.com/enlite-ai/maze/discussions) or [StackOverflow](https://stackoverflow.com/questions/tagged/maze-rl).
