# Frozen Lake Environment
  Model-based methods are tested on this environment in more of an [assignment](https://github.com/RaviAgrawal-1824/Assignment-1-Frozen-Lake) format.

## Table of Contents
- [Requirements](#requirements)
- [Description](#description)
- [State Space](#state-space)
- [Action Space](#action-space)
- [Reward Function](#reward-function)
- [Note](#note)

## Requirements
You need to install the following libraries to use this environment.
- `pip install gym` - Gym
- `pip install numpy` - NumPy
- `pip install matplotlib` - Matplotlib

## Description
Frozen lake involves crossing a frozen lake from Start(top left) to Goal(bottom right) without falling into any Holes by walking over the Frozen lake. The agent may not always move in the intended direction due to the slippery nature of the frozen lake.

| <img src="https://user-images.githubusercontent.com/109269344/226231759-ab497e8c-fc43-488d-a339-1a614637dc0c.gif" width="248" height="248" /> | <img src="https://user-images.githubusercontent.com/109269344/226231769-dec01bca-b92a-43b7-a5cd-6e28961bd4f2.gif" width="248" height="248" /> | <img src="https://user-images.githubusercontent.com/109269344/226231783-2d6cf9af-f8a5-4184-a299-e98b4efed5bc.gif" width="248" height="248" /> |
|:--:|:--:|:--:|
| `FrozenLake-v1`| `FrozenLake8x8-v1`| `FrozenLake-v1` |
|`is_slippery = False` |`is_slippery = False` |`is_slippery = True`|
| Deterministic Environment | Deterministic Environment | Stochastic Environment |

## State Space
Each map grid is represented by a number(current_row*max_columns + current_column). The State Space is dependent on the size of the map. For example, the 4x4 map has 16 possible states.

## Action Space
The agent can take 4 actions from any state which decides the direction to move in which can be:
- Left - 0
- Down - 1
- Right - 2
- Up - 3

## Reward Function
- When the agent reaches goal: +1
- When the falls into the hole: 0
- When the agent reaches a frozen path: 0

## Note
Custom maps can also be created for frozen lake environments. For more information on this environment, please see its [Gym Documentation](https://www.gymlibrary.dev/environments/toy_text/frozen_lake/#rewards).