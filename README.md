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
Tested on different FrozenLake environments. (link)

### Policy Iteration
This algorithm involves an iterative process that alternates between evaluating and improving the current policy until an optimal policy is found. The policy evaluation step is iterative until convergence then policy improvement step is taken.
The algorithm is implemented on three environments:
    "FrozenLake-v1", is_slippery = False 35 steps
    "FrozenLake8x8-v1" 135 steps
    "FrozenLake-v1", is_slippery = True 35 steps environment is stochastic

### Value Iteration
This algorithm also involves an iterative process that alternates between evaluating and improving the current policy but the policy evaluation step doesnot wait for its convergence and policy improvement step is taken. Evaluation and Improvement step is taken alternatively till optimal policy is found.
The algorithm is implemented on three environments:
    "FrozenLake-v1", is_slippery = False 15 steps
    "FrozenLake8x8-v1" 31 steps
    "FrozenLake-v1", is_slippery = True  15 steps environment is stochastic

 ------------gif 1, gif 2, gif 3
 
 ## Model Free Methods
  Model free algorithms do not require a model of the underlying environment. The agent learns the optimal policy or value function by interacting with the environment and observing rewards.

  ### Tabular Methods
    Tabular methods maintain a table of values for each state-action pair in the environment. The agent explores the environment and append new observations in the table and iteratively estimates its true value fuctions. They are effective in small to medium-sized environments.
    Tested on different Minigrid environments. (link)

#### SARSA
SARSA (State-Action-Reward-State-Action) learns to find the optimal policy by estimating the action-value function. The action value is estimated through Temporal Difference(TD-0) method bootstrapped from the next (state,action) pair. It has low variance but introduces comparatively higher bias.
This algorithm is implemented on four environments:
    "MiniGrid-Empty-6x6-v0"
    "MiniGrid-Empty-8x8-v0"
    "MiniGrid-Empty-Random-6x6-v0"
    "MiniGrid-Dynamic-Obstacles-6x6-v0"

#### SARSA-Lambda
In this algorithm, the action value is estimated as a geometrical weighted avergae over all time step returns. The estimate donot considers only the next state but all the intermediate steps till the terminal step. It is a more robust algorithm having low variance and low bias.
This algorithm is implemented on four environments:
    "MiniGrid-Empty-6x6-v0"
    "MiniGrid-Empty-8x8-v0"
    "MiniGrid-Empty-Random-6x6-v0"
    "MiniGrid-Dynamic-Obstacles-6x6-v0"

#### SARSA-Backwards
This algorithm uses Eligibility Traces which is based on frequency and recency heuristics. Eligibility traces are a way of assigning credit to (state,action) pairs that led to a particular reward. It helps in updating the values functions in proportion to the credit assigned to them. The updates to the states are equivalent to that in SARSA-Lambda at the end of the episode for offline training.
This algorithm is implemented on four environments:
    "MiniGrid-Empty-6x6-v0"
    "MiniGrid-Empty-8x8-v0"
    "MiniGrid-Empty-Random-6x6-v0"
    "MiniGrid-Dynamic-Obstacles-6x6-v0"

gifs and comparative graphs

#### Monte-Carlo 
This algorithm uses Monte-Carlo approach for action value estimation where the actual return is calculated by summing up the discounted rewards till the terminal step. This method has zero bias but has high variance.
The algorithm is implemented on four environments:
    "MiniGrid-Empty-6x6-v0"
    "MiniGrid-Empty-8x8-v0"
    "MiniGrid-Empty-Random-6x6-v0"
    "MiniGrid-Dynamic-Obstacles-6x6-v0"

#### Q-learning
This algorithm is an OFF-policy algorithm. Its behaviour policy is Epsilon-greedy whereas the target policy is fully greedy. The action values are bootstrapped from the best action value of the next state.
The algorithm is implemented on four environments:
    "MiniGrid-Empty-6x6-v0"
    "MiniGrid-Empty-8x8-v0"
    "MiniGrid-Empty-Random-6x6-v0"
    "MiniGrid-Dynamic-Obstacles-6x6-v0"

gifs and comparative graphs

#### effect on learning with variations in hyperparameters

### Approximation Methods

#### Action Value Function Approximation
#### Policy Gradient
