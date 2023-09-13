# Implementation_of_RL_Algorithms
 Implementation of different Reinforcement Learning Algorithms in various Gym environments.

## Table of Contents
- [Model-based Methods](#model-based-methods)
  - [Policy Iteration](#policy-iteration)
  - [Value Iteration](#value-iteration)
- [Model-free Methods](#model-free-methods)
  - [Tabular Methods](#tabular-methods)
     - [SARSA](#sarsa) 
     - [SARSA-Lambda](#sarsa-lambda) 
     - [SARSA-Backwards](#sarsa-backwards)
     - [Monte-Carlo](#monte-carlo) 
     - [Q-learning](#q-learning)
  - [Learning-based Methods](#learning-based-methods)
     - [Deep Q-Networks(DQN)](#deep-q-networks)
      - [Actor-Critic](#actor-critic)

## Model-Based Methods
Model-based algorithms require the complete dynamics of the environment for their implementation. The value function for each state is computed using the model of the environment and optimal policy is determined accordingly.
Tested on different [FrozenLake environments](https://www.gymlibrary.dev/environments/toy_text/frozen_lake/):
   1. `FrozenLake-v1`, `is_slippery = False`
   2. `FrozenLake8x8-v1`, `is_slippery = False`
   3. `FrozenLake-v1`, `is_slippery = True`

- ### Policy Iteration
This algorithm involves an iterative process that alternates between evaluating and improving the current policy until an optimal policy is found. The policy evaluation step is iterative until convergence then the policy improvement step is taken.

- ### Value Iteration
This algorithm also involves an iterative process that alternates between evaluating and improving the current policy but the policy evaluation step does not wait for its convergence and the policy improvement step is taken. Evaluation and Improvement step is taken alternatively till optimal policy is found.

| <img src="https://user-images.githubusercontent.com/109269344/226231759-ab497e8c-fc43-488d-a339-1a614637dc0c.gif" width="250" height="250" /> | <img src="https://user-images.githubusercontent.com/109269344/226231769-dec01bca-b92a-43b7-a5cd-6e28961bd4f2.gif" width="250" height="250" /> | <img src="https://user-images.githubusercontent.com/109269344/226231783-2d6cf9af-f8a5-4184-a299-e98b4efed5bc.gif" width="250" height="250" /> |
|:--:|:--:|:--:|
| `FrozenLake-v1`| `FrozenLake8x8-v1`| `FrozenLake-v1` |
|`is_slippery = False` |`is_slippery = False` |`is_slippery = True`|
| Policy Iteration took 35 Steps. | Policy Iteration took 135 Steps. | Policy Iteration took 35 Steps. |
| Value Iteration took 15 Steps. | Value Iteration took 31 Steps.  | Value Iteration took 15 Steps.  |
 
 ## Model-free Methods
  Model-free algorithms do not require a model of the underlying environment. The agent learns the optimal policy or value function by interacting with the environment and observing rewards.

  ### Tabular Methods
Tabular methods maintain a table of values for each state-action pair in the environment. The agent explores the environment and appends new observations in the table and iteratively estimates its true value functions. They are effective in small to medium-sized environments.
Tested on different [Minigrid environments](https://github.com/mit-acl/gym-minigrid):
   1. `MiniGrid-Empty-6x6-v0`
   2. `MiniGrid-Empty-8x8-v0`
   3. `MiniGrid-Empty-Random-6x6-v0`
   4. `MiniGrid-Dynamic-Obstacles-6x6-v0`

|<img src="https://user-images.githubusercontent.com/109269344/226203830-b526b6be-70f1-412c-aae6-83450dfc2c45.gif" width="180" height="200" /> | <img src="https://user-images.githubusercontent.com/109269344/226203835-46684f61-1d6e-4de1-a8c2-547d60561dc4.gif" width="180" height="200" /> | <img src="https://user-images.githubusercontent.com/109269344/226203845-121714a2-48b6-4f42-83d9-570ae74e2e44.gif" width="180" height="200" /> | <img src="https://user-images.githubusercontent.com/109269344/226203854-535a0f13-8098-4b0a-b988-bbb2104eeb65.gif" width="180" height="200" /> |
|:--:|:--:|:--:|:--:|
|`MiniGrid-Empty-6x6-v0`|`MiniGrid-Empty-8x8-v0`|`MiniGrid-Empty-Random-6x6-v0`|`MiniGrid-Dynamic-Obstacles-6x6-v0`|

- #### SARSA
SARSA (State-Action-Reward-State-Action) learns to find the optimal policy by estimating the action-value function. The action value is estimated through the Temporal Difference(TD-0) method bootstrapped from the next (state, action) pair. It has low variance but introduces comparatively higher bias.

- #### SARSA-Lambda
In this algorithm, the action value is estimated as a geometrically weighted average over all time step returns. The estimate does not consider only the next state but all the intermediate steps till the terminal step. It is a more robust algorithm having low variance and low bias.

- #### SARSA-Backwards
This algorithm uses Eligibility Traces which is based on frequency and recency heuristics. Eligibility traces are a way of assigning credit to (state, action) pairs that led to a particular reward. It helps in updating the value functions in proportion to the credit assigned to them. The updates to the states are equivalent to those in SARSA-Lambda at the end of the episode for offline training.
          
|<img src="https://user-images.githubusercontent.com/109269344/226196330-57027daa-29b5-4c4f-8fb0-3580fe43db19.png" width="370" height="350"/> | <img src="https://user-images.githubusercontent.com/109269344/226195527-411e47d4-b6e3-4dad-a3d0-d2d38e81207d.png" width="370" height="350"/>|
|:--:|:--:|
|<img src="https://user-images.githubusercontent.com/109269344/226196318-84549a3b-566c-42a8-887a-938c282e6d65.png" width="370" height="350"/> | <img src="https://user-images.githubusercontent.com/109269344/226195536-9e61799d-05ed-468c-8e88-cfba57ceb416.png" width="370" height="350"/>|
|<img src="https://user-images.githubusercontent.com/109269344/226196301-044e7f3c-881c-4456-b1da-01543e4a8c83.png" width="370" height="350"/> | <img src="https://user-images.githubusercontent.com/109269344/226195550-697a578e-c8b0-454e-bcc6-c84bf9978568.png" width="370" height="350"/>|
|<img src="https://user-images.githubusercontent.com/109269344/226196292-10baa217-f675-45c8-9781-6b37f52349d2.png" width="370" height="350"/> | <img src="https://user-images.githubusercontent.com/109269344/226195554-4380d934-a4f9-41cd-b842-3a292ae538c6.png" width="370" height="350"/>|

- #### Monte-Carlo 
This algorithm uses the Monte-Carlo approach for action value estimation where the actual return is calculated by summing up the discounted rewards till the terminal step. This method has zero bias but has high variance.

- #### Q-learning
This algorithm is an OFF-policy algorithm. Its behaviour policy is Epsilon-greedy whereas the target policy is fully greedy. The action values are bootstrapped from the best action value of the next state.

|<img src="https://user-images.githubusercontent.com/109269344/226198220-0475d499-6def-4ac8-8516-a2cb0711d09d.png" width="370" height="350" /> | <img src="https://user-images.githubusercontent.com/109269344/226198243-9e67044d-bede-4e96-a879-a28f580e469e.png" width="370" height="350"/>|
|:--:|:--:|
|<img src="https://user-images.githubusercontent.com/109269344/226198200-b08009a1-2194-43ce-a03d-2a5b4d16102a.png" width="370" height="350" /> | <img src="https://user-images.githubusercontent.com/109269344/226198202-d026b494-f37b-4442-b999-d20fcc14fbd8.png" width="370" height="350"/>|
|<img src="https://user-images.githubusercontent.com/109269344/226198209-e2759d2c-fe0e-4c3d-b524-36ec0ca05970.png" width="370" height="350" /> | <img src="https://user-images.githubusercontent.com/109269344/226198217-cff66e6c-ac87-41c1-95ae-1c443dcf99ce.png" width="370" height="350"/>|
|<img src="https://user-images.githubusercontent.com/109269344/226198029-6724bb6d-0cc0-4ce3-8e19-1b5ba7049238.png" width="370" height="350" /> | <img src="https://user-images.githubusercontent.com/109269344/226198033-3da95c42-4fb5-49eb-ba8c-2db5bf77fd3b.png" width="370" height="350"/>|

<!-- gifs and comparative graphs, hyper-parameters tuning

#### effect on learning with variations in hyperparameters -->

### Learning-based Methods
Learning-based methods involve directly learning the optimal policy or value function without constructing an explicit model of the environment. These methods utilize machine learning techniques, often neural networks, to approximate complex value functions or policies. These methods are especially suitable for environments with high-dimensional state spaces or continuous action spaces where tabular methods might be impractical. Tested on different [Gym environments](https://www.gymlibrary.dev/environments/classic_control/): 
   * `CartPole-v1`
   * `Pendulum-v1`

|<img src = "https://github.com/RaviAgrawal-1824/Implementation_of_RL_Algorithms/assets/109269344/f6d2123f-c767-4c4a-a4dd-613efb1c51a6" width="400" height="350"/> | <img src = "https://github.com/RaviAgrawal-1824/Implementation_of_RL_Algorithms/assets/109269344/162ecffa-38cf-467f-9252-4182880e1c73" width="400" height="350"/>|
|:--:|:--:|
|`CartPole-v1`|`Pendulum-v1`|

- #### Deep Q-Networks
DQN extends Q-learning by using neural networks to approximate the action-value function. It uses an experience-replay buffer to store the history and train the model to estimate the action-value function. Fixed Q-target is used to enhance the stability in it.  
Tested in `CartPole-v1`.

|<img width="370" height="350" src="https://github.com/RaviAgrawal-1824/Implementation_of_RL_Algorithms/assets/109269344/ade482d6-8cd0-4aa4-9848-2790a1e69920" />|<img width="370" height="350" src="https://github.com/RaviAgrawal-1824/Implementation_of_RL_Algorithms/assets/109269344/77a66ac1-860b-4b89-824c-19aadb79ae0e"/>|
|:--:|:--:|
<img width="370" height="350" src="https://github.com/RaviAgrawal-1824/Implementation_of_RL_Algorithms/assets/109269344/22acd915-9a3c-4bc9-b4a1-62dcb3aac91a"/>

- #### Actor-Critic
Actor-critic methods deploy neural networks for estimation of both policy(by Actor) and value function(by Critic). It is useful for environments with continuous action spaces. Tested in `Pendulum-v1`.
