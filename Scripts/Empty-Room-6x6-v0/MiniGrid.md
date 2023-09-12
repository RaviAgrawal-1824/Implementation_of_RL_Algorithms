# MiniGrid Empty Room Environment

## Table of Contents
- [Requirements](#requirements)
- [Description](#description)
- [State Space](#state-space)
- [Action Space](#action-space)
- [Reward Function](#reward-function)
- [Algorithms implemented](#algorithms-implemented)

## Requirements
You need to install the following libraries to use this environment.
- `pip install gymnasium` - Gymnasium
- `pip install minigrid` - Minigrid
- `pip install numpy` - NumPy
- `pip install matplotlib` - Matplotlib

## Description
This environment is an empty room, and the goal of the agent is to reach the green goal square at the bottom right. The random variants of the environment have the agent starting at a random position for each episode, while the regular variants have the agent always starting in the top-left corner.

|<img src="https://user-images.githubusercontent.com/109269344/226203830-b526b6be-70f1-412c-aae6-83450dfc2c45.gif" width="248" height="248" /> | <img src="https://user-images.githubusercontent.com/109269344/226203835-46684f61-1d6e-4de1-a8c2-547d60561dc4.gif" width="248" height="248" /> | <img src="https://user-images.githubusercontent.com/109269344/226203845-121714a2-48b6-4f42-83d9-570ae74e2e44.gif" width="248" height="248" /> |
|:--:|:--:|:--:|
|`MiniGrid-Empty-6x6-v0`|`MiniGrid-Empty-8x8-v0`|`MiniGrid-Empty-Random-6x6-v0`|

## State Space
Observation Space includes `image: Box(7, 7, 3)` which corresponds to the agent's view encoded as a 3-dimensional tuple: (OBJECT_IDX, COLOR_IDX, STATE). This 3D tuple can be flattened into a 1D tuple and can be used as a key in a dictionary for tabular methods.  
Alternatively, `env.agent_pos` clubbed with `env.agent_dir` is sufficient to make the state space Markov and can be used as State Space representation.  
The State Space is dependent on the size of the room. For example, the 4x4 room has 64 possible states(although the name says 6x6).

## Action Space
The agent can take 7 actions from any state in all MiniGrid environments:
- 0 - Left
- 1 - Right
- 2 - Forward
- 3 - Pickup (Not used)
- 4 - Drop (Not used)
- 5 - Toggle (Not used)
- 6 - Done (Not used)

## Reward Function
The goal of the agent is to reach the green goal square, which provides a sparse reward. A small penalty is subtracted for the number of steps to reach the goal.  
A reward of ‘1 - 0.9 * (step_count / max_steps)’ is given for success, and ‘0’ for failure.

## Algorithms implemented
Tabular Model-Free algorithms are tested on this environment like:
- Monte-Carlo
- SARSA
- SARSA Lambda
- SARSA Backwards
- Q-Learning


For more information on this environment, please see its [MiniGrid Documentation](https://minigrid.farama.org/environments/minigrid/EmptyEnv/).
