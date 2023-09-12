# MiniGrid Dynamic Obstacles Environment

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
This environment is an empty room with moving obstacles. The goal of the agent is to reach the green goal square without colliding with any obstacle. The random variants of the environment have the agent starting at a random position for each episode, while the regular variants have the agent always starting in the top-left corner.

|<img src="https://user-images.githubusercontent.com/109269344/226203854-535a0f13-8098-4b0a-b988-bbb2104eeb65.gif" width="300" height="300" /> |
|:--:|
|`MiniGrid-Dynamic-Obstacles-6x6-v0`|

## State Space
Observation Space includes `image: Box(7, 7, 3)` which corresponds to the agent's view encoded as a 3-dimensional tuple: (OBJECT_IDX, COLOR_IDX, STATE). This 3D tuple can be flattened into a 1D tuple and can be used as a key in a dictionary for tabular methods.  
Alternatively, `env.agent_pos`, `env.agent_dir`and presence of obstacles around agent is sufficient to make the state space Markov and can be used as State Space representation.  
The State Space is dependent on the size of the room. For example, the 4x4 room has 512 possible states using 2nd representation (although the name says 6x6).

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
The goal of the agent is to reach the green goal square without colliding with any obstacle. A large penalty is subtracted if the agent collides with an obstacle and the episode finishes. A small penalty is subtracted for the number of steps to reach the goal.  
- If agent collides with obstacles, reward = -1
- If agent reaches successfully, reward = 1 - 0.9 * (step_count / max_steps)
- If agent fails, reward = 0

## Algorithms implemented
Tabular Model-Free algorithms are tested on this environment like:
- SARSA Lambda
- SARSA Backwards
- Q-Learning


For more information on this environment, please see its [MiniGrid Documentation](https://minigrid.farama.org/environments/minigrid/DynamicObstaclesEnv/).
