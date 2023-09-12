# Cart Pole Environment

## Table of Contents
- [Requirements](#requirements)
- [Description](#description)
- [State Space](#state-space)
- [Action Space](#action-space)
- [Reward Function](#reward-function)
- [Algorithms implemented](#algorithms-implemented)

## Requirements
You need to install the following libraries to use this environment.
- `pip install gym` - Gym
- `pip install numpy` - NumPy
- `pip install torch` - PyTorch
- `pip install matplotlib` - Matplotlib

## Description
A pole is attached by a hinge to a cart, which moves along a frictionless track. The pendulum is placed upright on the cart and the goal is to balance the pole by applying forces in the left and right direction on the cart.

|<img src="https://github.com/RaviAgrawal-1824/Implementation_of_RL_Algorithms/assets/109269344/f6d2123f-c767-4c4a-a4dd-613efb1c51a6" width="300" height="300" /> |
|:--:|
|`CartPole-v1`|

## State Space
Observation Space includes a ndarray with shape (4,) which corresponds to (Cart position, Cart velocity, Pole angle, Pole angular velocity). All the values are continuous which makes the state space quite huge.
- Although The cart x-position can take values between (-4.8, 4.8), but the episode terminates if the cart leaves the (-2.4, 2.4) range.
- The pole angle can be observed between (-.418, .418) radians (or ±24°), but the episode terminates if the pole angle is not in the range (-.2095, .2095) (or ±12°).

## Action Space
The agent can take 2 actions from any state which indicates the direction of the fixed force the cart is pushed with.
- 0 - Push the cart to the left
- 1 - Push the cart to the right

## Reward Function
A reward of +1 is given for every step, the pole is in the required range (including the termination step). The threshold for rewards is 475.

## Algorithms implemented
Learning based Model-Free algorithms are tested on this environment like:
- DQN(Deep-Q-Network)

For more information on this environment, please see its [Gym Documentation](https://www.gymlibrary.dev/environments/classic_control/cart_pole/).
