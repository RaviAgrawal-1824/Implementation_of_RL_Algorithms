# Implementation_of_RL_Algorithms
 Implementation of different Reinforcement Learning Algorithms on various Gym Environmnets.

## Table of Contents
- [Model Based Methods](#model-based-methods)
  - [Policy Iteration](#policy-iteration)
  - [Value Iteration](#value-iteration)
- [Model Free Methods](#model-free-methods)
  - [Tabular Methods](#tabular-methods)
     - [SARSA](#sarsa) 
     - [SARSA-Lambda](#sarsa-lambda) 
     - [SARSA-Backwards](#sarsa-backwards)
     - [Monte-Carlo](#monte-carlo) 
     - [Q-learning](#q-learning)
  - [Approximation Methods](#approximation-methods)
     - [Action Value Function Approximation](#action-value-function-approximation)
      - [Policy Gradient](#policy-gradient)

## Model Based Methods
Model Based algorithms require the complete dynamics of the environment for their implemetation. Value function for each state is computed using the model of the environment and optimal policy is determined accordingly.
Tested on different [FrozenLake environments](https://www.gymlibrary.dev/environments/toy_text/frozen_lake/).

### Policy Iteration
This algorithm involves an iterative process that alternates between evaluating and improving the current policy until an optimal policy is found. The policy evaluation step is iterative until convergence then policy improvement step is taken.
The algorithm is implemented on three environments:
   * `FrozenLake-v1`, `is_slippery = False`
   * `FrozenLake8x8-v1`, `is_slippery = False`
   * `FrozenLake-v1`, `is_slippery = True`

### Value Iteration
This algorithm also involves an iterative process that alternates between evaluating and improving the current policy but the policy evaluation step doesnot wait for its convergence and policy improvement step is taken. Evaluation and Improvement step is taken alternatively till optimal policy is found.
The algorithm is implemented on three environments:
   * `FrozenLake-v1`, `is_slippery = False`
   * `FrozenLake8x8-v1`, `is_slippery = False`
   * `FrozenLake-v1`, `is_slippery = True`

| <img src="https://user-images.githubusercontent.com/109269344/226192078-5076c582-ba7e-41c8-ad31-b3ea148db654.gif" width="300" height="300" /> | <img src="https://user-images.githubusercontent.com/109269344/226192106-7dade730-5cbf-42ea-9b9f-9c6df4a46a8b.gif" width="300" height="300" /> | <img src="https://user-images.githubusercontent.com/109269344/226192214-3e66c9ed-25a5-45ad-ac8f-723e5f54aa9d.gif" width="300" height="300" /> |
|:--:|:--:|:--:|
| `FrozenLake-v1`,`is_slippery = False` | `FrozenLake8x8-v1`,`is_slippery = False` | `FrozenLake-v1`, `is_slippery = True`|
| Policy Iteration = 35 Steps | Policy Iteration = 135 Steps | Policy Iteration = 35 Steps |
| Value Iteration = 15 Steps | Value Iteration = 31 Steps  | Value Iteration = 15 Steps  |
 
 ## Model Free Methods
  Model free algorithms do not require a model of the underlying environment. The agent learns the optimal policy or value function by interacting with the environment and observing rewards.

  ### Tabular Methods
Tabular methods maintain a table of values for each state-action pair in the environment. The agent explores the environment and append new observations in the table and iteratively estimates its true value fuctions. They are effective in small to medium-sized environments.
Tested on different [Minigrid environments](https://github.com/mit-acl/gym-minigrid):
   * `MiniGrid-Empty-6x6-v0`
   * `MiniGrid-Empty-8x8-v0`
   * `MiniGrid-Empty-Random-6x6-v0`
   * `MiniGrid-Dynamic-Obstacles-6x6-v0`

|<img src="https://user-images.githubusercontent.com/109269344/226203830-b526b6be-70f1-412c-aae6-83450dfc2c45.gif" width="225" height="225" /> | <img src="https://user-images.githubusercontent.com/109269344/226203835-46684f61-1d6e-4de1-a8c2-547d60561dc4.gif" width="225" height="225" /> | <img src="https://user-images.githubusercontent.com/109269344/226203845-121714a2-48b6-4f42-83d9-570ae74e2e44.gif" width="225" height="225" /> | <img src="https://user-images.githubusercontent.com/109269344/226203854-535a0f13-8098-4b0a-b988-bbb2104eeb65.gif" width="225" height="225" /> |
|:--:|:--:|:--:|:--:|
|`MiniGrid-Empty-6x6-v0`|`MiniGrid-Empty-8x8-v0`|`MiniGrid-Empty-Random-6x6-v0`|`MiniGrid-Dynamic-Obstacles-6x6-v0`|

#### SARSA
SARSA (State-Action-Reward-State-Action) learns to find the optimal policy by estimating the action-value function. The action value is estimated through Temporal Difference(TD-0) method bootstrapped from the next (state,action) pair. It has low variance but introduces comparatively higher bias.
This algorithm is implemented on four environments:
   * `MiniGrid-Empty-6x6-v0`
   * `MiniGrid-Empty-8x8-v0`
   * `MiniGrid-Empty-Random-6x6-v0`
   * `MiniGrid-Dynamic-Obstacles-6x6-v0`

#### SARSA-Lambda
In this algorithm, the action value is estimated as a geometrical weighted avergae over all time step returns. The estimate donot considers only the next state but all the intermediate steps till the terminal step. It is a more robust algorithm having low variance and low bias.
This algorithm is implemented on four environments:
   * `MiniGrid-Empty-6x6-v0`
   * `MiniGrid-Empty-8x8-v0`
   * `MiniGrid-Empty-Random-6x6-v0`
   * `MiniGrid-Dynamic-Obstacles-6x6-v0`

#### SARSA-Backwards
This algorithm uses Eligibility Traces which is based on frequency and recency heuristics. Eligibility traces are a way of assigning credit to (state,action) pairs that led to a particular reward. It helps in updating the values functions in proportion to the credit assigned to them. The updates to the states are equivalent to that in SARSA-Lambda at the end of the episode for offline training.
This algorithm is implemented on four environments:
   * `MiniGrid-Empty-6x6-v0`
   * `MiniGrid-Empty-8x8-v0`
   * `MiniGrid-Empty-Random-6x6-v0`
   * `MiniGrid-Dynamic-Obstacles-6x6-v0`
          
<img src="https://user-images.githubusercontent.com/109269344/226196330-57027daa-29b5-4c4f-8fb0-3580fe43db19.png" width="480" height="400" align="left" hspace="10"/><img src="https://user-images.githubusercontent.com/109269344/226195527-411e47d4-b6e3-4dad-a3d0-d2d38e81207d.png" width="480" height="400"/>  

<img src="https://user-images.githubusercontent.com/109269344/226196318-84549a3b-566c-42a8-887a-938c282e6d65.png" width="480" height="400" align="left" hspace="10"/><img src="https://user-images.githubusercontent.com/109269344/226195536-9e61799d-05ed-468c-8e88-cfba57ceb416.png" width="480" height="400"/> 

<img src="https://user-images.githubusercontent.com/109269344/226196301-044e7f3c-881c-4456-b1da-01543e4a8c83.png" width="480" height="400" align="left" hspace="10"/><img src="https://user-images.githubusercontent.com/109269344/226195550-697a578e-c8b0-454e-bcc6-c84bf9978568.png" width="480" height="400"/> 

<img src="https://user-images.githubusercontent.com/109269344/226196292-10baa217-f675-45c8-9781-6b37f52349d2.png" width="480" height="400" align="left" hspace="10"/><img src="https://user-images.githubusercontent.com/109269344/226195554-4380d934-a4f9-41cd-b842-3a292ae538c6.png" width="480" height="400"/>

#### Monte-Carlo 
This algorithm uses Monte-Carlo approach for action value estimation where the actual return is calculated by summing up the discounted rewards till the terminal step. This method has zero bias but has high variance.
The algorithm is implemented on four environments:
   * `MiniGrid-Empty-6x6-v0`
   * `MiniGrid-Empty-8x8-v0`
   * `MiniGrid-Empty-Random-6x6-v0`
   * `MiniGrid-Dynamic-Obstacles-6x6-v0`

#### Q-learning
This algorithm is an OFF-policy algorithm. Its behaviour policy is Epsilon-greedy whereas the target policy is fully greedy. The action values are bootstrapped from the best action value of the next state.
The algorithm is implemented on four environments:
   * `MiniGrid-Empty-6x6-v0`
   * `MiniGrid-Empty-8x8-v0`
   * `MiniGrid-Empty-Random-6x6-v0`
   * `MiniGrid-Dynamic-Obstacles-6x6-v0`

<img src="https://user-images.githubusercontent.com/109269344/226198220-0475d499-6def-4ac8-8516-a2cb0711d09d.png" width="480" height="400" align="left" hspace="10"/><img src="https://user-images.githubusercontent.com/109269344/226198243-9e67044d-bede-4e96-a879-a28f580e469e.png" width="480" height="400"/>  

<img src="https://user-images.githubusercontent.com/109269344/226198200-b08009a1-2194-43ce-a03d-2a5b4d16102a.png" width="480" height="400" align="left" hspace="10"/><img src="https://user-images.githubusercontent.com/109269344/226198202-d026b494-f37b-4442-b999-d20fcc14fbd8.png" width="480" height="400"/> 

<img src="https://user-images.githubusercontent.com/109269344/226198209-e2759d2c-fe0e-4c3d-b524-36ec0ca05970.png" width="480" height="400" align="left" hspace="10"/><img src="https://user-images.githubusercontent.com/109269344/226198217-cff66e6c-ac87-41c1-95ae-1c443dcf99ce.png" width="480" height="400"/> 

<img src="https://user-images.githubusercontent.com/109269344/226198029-6724bb6d-0cc0-4ce3-8e19-1b5ba7049238.png" width="480" height="400" align="left" hspace="10"/><img src="https://user-images.githubusercontent.com/109269344/226198033-3da95c42-4fb5-49eb-ba8c-2db5bf77fd3b.png" width="480" height="400"/>

gifs and comparative graphs, hyper-parameters tuning

#### effect on learning with variations in hyperparameters

### Approximation Methods

#### Action Value Function Approximation
#### Policy Gradient
