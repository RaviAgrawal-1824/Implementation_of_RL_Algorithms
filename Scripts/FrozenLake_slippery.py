# This slippery environment is stochastic i.e., on taking same action from the same state, the resulatant state is not fixed. 
# Environemnet can blow us to different states with some probability defined in the dynamics of the environment. 
# The outcome of an action taken by the agent can be uncertain, and may depend on random factors that are not under the control of the agent.

import numpy as np
import gym
import time

"""
For policy_evaluation, policy_improvement, policy_iteration and value_iteration,
the parameters P, nS, nA, gamma are defined as follows:

    P: nested dictionary
        From gym.core.Environment
        For each pair of states in [1, nS] and actions in [1, nA], P[state][action] is a
        tuple of the form (probability, nextstate, reward, terminal) where
            - probability: float
                the probability of transitioning from "state" to "nextstate" with "action"
            - nextstate: int
                denotes the state we transition to (in range [0, nS - 1])
            - reward: int
                either 0 or 1, the reward for transitioning from "state" to
                "nextstate" with "action"
            - terminal: bool
                True when "nextstate" is a terminal state (hole or goal), False otherwise
    nS: int
        number of states in the environment
    nA: int
        number of actions in the environment
    gamma: float
        Discount factor. Number in range [0, 1)
"""

def policy_evaluation(P, nS, nA, policy, gamma=0.9, tol=1e-3):
    """Evaluate the value function from a given policy.
        Parameters
        ----------
        P, nS, nA, gamma:
            defined at beginning of file
        policy: np.array[nS]
            The policy to evaluate. Maps states to actions.
        tol: float
            Terminate policy evaluation when
                max |value_function(s) - prev_value_function(s)| < tol
        Returns
        -------
        value_function: np.ndarray[nS]
            The value function of the given policy, where value_function[s] is
            the value of state s
        """

    value_function = np.zeros(nS)
    ############################
    # YOUR IMPLEMENTATION HERE #
    
    while True:
        diff_list=[]
        for i in range(nS):
            prev=0
            diff=value_function[i]
            for j in P[i][policy[i]]:    # looping over all the possible final states in which the environment might takes us in, for the same action in the same state.
                prev = prev + j[0]*(j[2] + value_function[j[1]])
            value_function[i] = gamma*prev
            diff_list.append(abs(value_function[i]-diff))
        if(max(diff_list)<tol):
            break

    ############################
    return value_function


def policy_improvement(P, nS, nA, value_from_policy, policy, gamma=0.9):
    """Given the value function from policy improve the policy.

        Parameters
        ----------
        P, nS, nA, gamma:
            defined at beginning of file
        value_from_policy: np.ndarray
            The value calculated from the policy
        policy: np.array
            The previous policy.

        Returns
        -------
        new_policy: np.ndarray[nS]
            An array of integers. Each integer is the optimal action to take
            in that state according to the environment dynamics and the
            given value function.
        """
    new_policy = np.zeros(nS, dtype='int')

    ############################
    # YOUR IMPLEMENTATION HERE #

    for i in range(nS):
        q_pi=[]
        for j in range(nA):
            prev=0
            for k in P[i][j]:    # p[state][action] is a list of size 3 having 3 tuples each of length 4, 'in' keyword will return those tuples till they are available (i.e. 3 times).
                prev = prev + k[0] * (k[2] + value_from_policy[k[1]])    # multiplying reward and next possible states with their corr. possibilities and summing over them.
            q_pi.append(gamma*prev)     # appending the action value for each available action from that state.
        new_policy[i]=np.argmax(q_pi)    # argmax will return the index where the maximum value of the lis is stored, it will be the most suitable action to be taken thus integrated in the updated policy.

    ############################
    return new_policy


def policy_iteration(P, nS, nA, gamma=0.9, tol=10e-3):
    """Runs policy iteration.

        You should call the policy_evaluation() and policy_improvement() methods to
        implement this method.

        Parameters
        ----------
        P, nS, nA, gamma:
            defined at beginning of file
        tol: float
            tol parameter used in policy_evaluation()
        Returns:
        ----------
        value_function: np.ndarray[nS]
        policy: np.ndarray[nS]
        """
    value_function = np.zeros(nS)
    policy = np.zeros(nS, dtype=int)

    ############################
    # YOUR IMPLEMENTATION HERE #

    while True:
        value_function=policy_evaluation(P,nS,nA,policy,gamma,tol)
        prev_policy=policy
        policy=policy_improvement(P,nS,nA,value_function,prev_policy,gamma)
        if(prev_policy==policy).all():
            print(policy)
            break

    ############################

    return value_function, policy


def value_iteration(P, nS, nA, gamma=0.9, tol=1e-3):
    """
        Learn value function and policy by using value iteration method for a given
        gamma and environment.
        Parameters:
        ----------
        P, nS, nA, gamma:
            defined at beginning of file
        tol: float
            Terminate value iteration when
                max |value_function(s) - prev_value_function(s)| < tol
        Returns:
        ----------
        value_function: np.ndarray[nS]
        policy: np.ndarray[nS]
        """
    value_function = np.zeros(nS)
    policy = np.zeros(nS, dtype=int)

    ############################
    # YOUR IMPLEMENTATION HERE #
    # need to add the count as in the very first iteration, no major changes are there in the value function and the loop gets teminated in the first iteration only.
    # As the default null policy commands to go left from every state and going left has reward 0 from every state and the value function is already null for all state, so new value function for all states remains still 0 till the policy gets improved further.  
    count=0
    while True:
        count+=1
        diff_list=[]
        for i in range(nS):
            prev=0
            diff=value_function[i]
            for j in P[i][policy[i]]:
                prev = prev + j[0] * (j[2] + value_function[j[1]])
            value_function[i] = gamma*prev
            diff_list.append(abs(value_function[i]-diff))
        if(max(diff_list)<tol and count!=1):
            print(policy)
            break
        else:
            for i in range(nS):
                q_pi=[]
                for j in range(nA):
                    prev=0
                    for k in P[i][j]:
                        prev = prev + k[0]*(k[2] + value_function[k[1]])
                    q_pi.append(gamma*prev)
                policy[i]=np.argmax(q_pi)

    ############################
    return value_function, policy


def render_single(env, policy, max_steps=100):
    """
      This function does not need to be modified
      Renders policy once on environment. Watch your agent play!

      Parameters
      ----------
      env: gym.core.Environment
        Environment to play on. Must have nS, nA, and P as
        attributes.
      Policy: np.array of shape [env.nS]
        The action to take at a given state
    """
    episode_reward = 0
    ob = env.reset()
    ob=ob[0]
    for t in range(max_steps):
        env.render()
        time.sleep(0.25)
        a = policy[ob]
        ob, rew, done, _ = env.step(a)[0:4]
        episode_reward += rew
        if done:
            break
    env.render()
    if not done:
        print("The agent didn't reach a terminal state in {} steps.".format(max_steps))
    else:
        print("Episode reward: %f" % episode_reward)


if __name__ == "__main__":
    #name = "FrozenLake8x8-v1"
    name = "FrozenLake-v1"
    is_slippery = True
    if(name == "FrozenLake8x8-v1"):
        P={0: {0: [(1.0, 0, 0.0, False)], 1: [(1.0, 8, 0.0, False)], 2: [(1.0, 1, 0.0, False)], 3: [(1.0, 0, 0.0, False)]}, 1: {0: [(1.0, 0, 0.0, False)], 1: [(1.0, 9, 0.0, False)], 2: [(1.0, 2, 0.0, False)], 3: [(1.0, 1, 0.0, False)]}, 2: {0: [(1.0, 1, 0.0, False)], 1: [(1.0, 10, 0.0, False)], 2: [(1.0, 3, 0.0, False)], 3: [(1.0, 2, 0.0, False)]}, 3: {0: [(1.0, 2, 0.0, False)], 1: [(1.0, 11, 0.0, False)], 2: [(1.0, 4, 0.0, False)], 3: [(1.0, 3, 0.0, False)]}, 4: {0: [(1.0, 3, 0.0, False)], 1: [(1.0, 12, 0.0, False)], 2: [(1.0, 5, 0.0, False)], 3: [(1.0, 4, 0.0, False)]}, 5: {0: [(1.0, 4, 0.0, False)], 1: [(1.0, 13, 0.0, False)], 2: [(1.0, 6, 0.0, False)], 3: [(1.0, 5, 0.0, False)]}, 6: {0: [(1.0, 5, 0.0, False)], 1: [(1.0, 14, 0.0, False)], 2: [(1.0, 7, 0.0, False)], 3: [(1.0, 6, 0.0, False)]}, 7: {0: [(1.0, 6, 0.0, False)], 1: [(1.0, 15, 0.0, False)], 2: [(1.0, 7, 0.0, False)], 3: [(1.0, 7, 0.0, False)]}, 8: {0: [(1.0, 8, 0.0, False)], 1: [(1.0, 16, 0.0, False)], 2: [(1.0, 9, 0.0, False)], 3: [(1.0, 0, 0.0, False)]}, 9: {0: [(1.0, 8, 0.0, False)], 1: [(1.0, 17, 0.0, False)], 2: [(1.0, 10, 0.0, False)], 3: [(1.0, 1, 0.0, False)]}, 10: {0: [(1.0, 9, 0.0, False)], 1: [(1.0, 18, 0.0, False)], 2: [(1.0, 11, 0.0, False)], 3: [(1.0, 2, 0.0, False)]}, 11: {0: [(1.0, 10, 0.0, False)], 1: [(1.0, 19, 0.0, True)], 2: [(1.0, 12, 0.0, False)], 3: [(1.0, 3, 0.0, False)]}, 12: {0: [(1.0, 11, 0.0, False)], 1: [(1.0, 20, 0.0, False)], 2: [(1.0, 13, 0.0, False)], 3: [(1.0, 4, 0.0, False)]}, 13: {0: [(1.0, 12, 0.0, False)], 1: [(1.0, 21, 0.0, False)], 2: [(1.0, 14, 0.0, False)], 3: [(1.0, 5, 0.0, False)]}, 14: {0: [(1.0, 13, 0.0, False)], 1: [(1.0, 22, 0.0, False)], 2: [(1.0, 15, 0.0, False)], 3: [(1.0, 6, 0.0, False)]}, 15: {0: [(1.0, 14, 0.0, False)], 1: [(1.0, 23, 0.0, False)], 2: [(1.0, 15, 0.0, False)], 3: [(1.0, 7, 0.0, False)]}, 16: {0: [(1.0, 16, 0.0, False)], 1: [(1.0, 24, 0.0, False)], 2: [(1.0, 17, 0.0, False)], 3: [(1.0, 8, 0.0, False)]}, 17: {0: [(1.0, 16, 0.0, False)], 1: [(1.0, 25, 0.0, False)], 2: [(1.0, 18, 0.0, False)], 3: [(1.0, 9, 0.0, False)]}, 18: {0: [(1.0, 17, 0.0, False)], 1: [(1.0, 26, 0.0, False)], 2: [(1.0, 19, 0.0, True)], 3: [(1.0, 10, 0.0, False)]}, 19: {0: [(1.0, 19, 0, True)], 1: [(1.0, 19, 0, True)], 2: [(1.0, 19, 0, True)], 3: [(1.0, 19, 0, True)]}, 20: {0: [(1.0, 19, 0.0, True)], 1: [(1.0, 28, 0.0, False)], 2: [(1.0, 21, 0.0, False)], 3: [(1.0, 12, 0.0, False)]}, 21: {0: [(1.0, 20, 0.0, False)], 1: [(1.0, 29, 0.0, True)], 2: [(1.0, 22, 0.0, False)], 3: [(1.0, 13, 0.0, False)]}, 22: {0: [(1.0, 21, 0.0, False)], 1: [(1.0, 30, 0.0, False)], 2: [(1.0, 23, 0.0, False)], 3: [(1.0, 14, 0.0, False)]}, 23: {0: [(1.0, 22, 0.0, False)], 1: [(1.0, 31, 0.0, False)], 2: [(1.0, 23, 0.0, False)], 3: [(1.0, 15, 0.0, False)]}, 24: {0: [(1.0, 24, 0.0, False)], 1: [(1.0, 32, 0.0, False)], 2: [(1.0, 25, 0.0, False)], 3: [(1.0, 16, 0.0, False)]}, 25: {0: [(1.0, 24, 0.0, False)], 1: [(1.0, 33, 0.0, False)], 2: [(1.0, 26, 0.0, False)], 3: [(1.0, 17, 0.0, False)]}, 26: {0: [(1.0, 25, 0.0, False)], 1: [(1.0, 34, 0.0, False)], 2: [(1.0, 27, 0.0, False)], 3: [(1.0, 18, 0.0, False)]}, 27: {0: [(1.0, 26, 0.0, False)], 1: [(1.0, 35, 0.0, True)], 2: [(1.0, 28, 0.0, False)], 3: [(1.0, 19, 0.0, True)]}, 28: {0: [(1.0, 27, 0.0, False)], 1: [(1.0, 36, 0.0, False)], 2: [(1.0, 29, 0.0, True)], 3: [(1.0, 20, 0.0, False)]}, 29: {0: [(1.0, 29, 0, True)], 1: [(1.0, 29, 0, True)], 2: [(1.0, 29, 0, True)], 3: [(1.0, 29, 0, True)]}, 30: {0: [(1.0, 29, 0.0, True)], 1: [(1.0, 38, 0.0, False)], 2: [(1.0, 31, 0.0, False)], 3: [(1.0, 22, 0.0, False)]}, 31: {0: [(1.0, 30, 0.0, False)], 1: [(1.0, 39, 0.0, False)], 2: [(1.0, 31, 0.0, False)], 3: [(1.0, 23, 0.0, False)]}, 32: {0: [(1.0, 32, 0.0, False)], 1: [(1.0, 40, 0.0, False)], 2: [(1.0, 33, 0.0, False)], 3: [(1.0, 24, 0.0, False)]}, 33: {0: [(1.0, 32, 0.0, False)], 1: [(1.0, 41, 0.0, True)], 2: [(1.0, 34, 0.0, False)], 3: [(1.0, 25, 0.0, False)]}, 34: {0: [(1.0, 33, 0.0, False)], 1: [(1.0, 42, 0.0, True)], 2: [(1.0, 35, 0.0, True)], 3: [(1.0, 26, 0.0, False)]}, 35: {0: [(1.0, 35, 0, True)], 1: [(1.0, 35, 0, True)], 2: [(1.0, 35, 0, True)], 3: [(1.0, 35, 0, True)]}, 36: {0: [(1.0, 35, 0.0, True)], 1: [(1.0, 44, 0.0, False)], 2: [(1.0, 37, 0.0, False)], 3: [(1.0, 28, 0.0, False)]}, 37: {0: [(1.0, 36, 0.0, False)], 1: [(1.0, 45, 0.0, False)], 2: [(1.0, 38, 0.0, False)], 3: [(1.0, 29, 0.0, True)]}, 38: {0: [(1.0, 37, 0.0, False)], 1: [(1.0, 46, 0.0, True)], 2: [(1.0, 39, 0.0, False)], 3: [(1.0, 30, 0.0, False)]}, 39: {0: [(1.0, 38, 0.0, False)], 1: [(1.0, 47, 0.0, False)], 2: [(1.0, 39, 0.0, False)], 3: [(1.0, 31, 0.0, False)]}, 40: {0: [(1.0, 40, 0.0, False)], 1: [(1.0, 48, 0.0, False)], 2: [(1.0, 41, 0.0, True)], 3: [(1.0, 32, 0.0, False)]}, 41: {0: [(1.0, 41, 0, True)], 1: [(1.0, 41, 0, True)], 2: [(1.0, 41, 0, True)], 3: [(1.0, 41, 0, True)]}, 42: {0: [(1.0, 42, 0, True)], 1: [(1.0, 42, 0, True)], 2: [(1.0, 42, 0, True)], 3: [(1.0, 42, 0, True)]}, 43: {0: [(1.0, 42, 0.0, True)], 1: [(1.0, 51, 0.0, False)], 2: [(1.0, 44, 0.0, False)], 3: [(1.0, 35, 0.0, True)]}, 44: {0: [(1.0, 43, 0.0, False)], 1: [(1.0, 52, 0.0, True)], 2: [(1.0, 45, 0.0, False)], 3: [(1.0, 36, 0.0, False)]}, 45: {0: [(1.0, 44, 0.0, False)], 1: [(1.0, 53, 0.0, False)], 2: [(1.0, 46, 0.0, True)], 3: [(1.0, 37, 0.0, False)]}, 46: {0: [(1.0, 46, 0, True)], 1: [(1.0, 46, 0, True)], 2: [(1.0, 46, 0, True)], 3: [(1.0, 46, 0, True)]}, 47: {0: [(1.0, 46, 0.0, True)], 1: [(1.0, 55, 0.0, False)], 2: [(1.0, 47, 0.0, False)], 3: [(1.0, 39, 0.0, False)]}, 48: {0: [(1.0, 48, 0.0, False)], 1: [(1.0, 56, 0.0, False)], 2: [(1.0, 49, 0.0, True)], 3: [(1.0, 40, 0.0, False)]}, 49: {0: [(1.0, 49, 0, True)], 1: [(1.0, 49, 0, True)], 2: [(1.0, 49, 0, True)], 3: [(1.0, 49, 0, True)]}, 50: {0: [(1.0, 49, 0.0, True)], 1: [(1.0, 58, 0.0, False)], 2: [(1.0, 51, 0.0, False)], 3: [(1.0, 42, 0.0, True)]}, 51: {0: [(1.0, 50, 0.0, False)], 1: [(1.0, 59, 0.0, True)], 2: [(1.0, 52, 0.0, True)], 3: [(1.0, 43, 0.0, False)]}, 52: {0: [(1.0, 52, 0, True)], 1: [(1.0, 52, 0, True)], 2: [(1.0, 52, 0, True)], 3: [(1.0, 52, 0, True)]}, 53: {0: [(1.0, 52, 0.0, True)], 1: [(1.0, 61, 0.0, False)], 2: [(1.0, 54, 0.0, True)], 3: [(1.0, 45, 0.0, False)]}, 54: {0: [(1.0, 54, 0, True)], 1: [(1.0, 54, 0, True)], 2: [(1.0, 54, 0, True)], 3: [(1.0, 54, 0, True)]}, 55: {0: [(1.0, 54, 0.0, True)], 1: [(1.0, 63, 1.0, True)], 2: [(1.0, 55, 0.0, False)], 3: [(1.0, 47, 0.0, False)]}, 56: {0: [(1.0, 56, 0.0, False)], 1: [(1.0, 56, 0.0, False)], 2: [(1.0, 57, 0.0, False)], 3: [(1.0, 48, 0.0, False)]}, 57: {0: [(1.0, 56, 0.0, False)], 1: [(1.0, 57, 0.0, False)], 2: [(1.0, 58, 0.0, False)], 3: [(1.0, 49, 0.0, True)]}, 58: {0: [(1.0, 57, 0.0, False)], 1: [(1.0, 58, 0.0, False)], 2: [(1.0, 59, 0.0, True)], 3: [(1.0, 50, 0.0, False)]}, 59: {0: [(1.0, 59, 0, True)], 1: [(1.0, 59, 0, True)], 2: [(1.0, 59, 0, True)], 3: [(1.0, 59, 0, True)]}, 60: {0: [(1.0, 59, 0.0, True)], 1: [(1.0, 60, 0.0, False)], 2: [(1.0, 61, 0.0, False)], 3: [(1.0, 52, 0.0, True)]}, 61: {0: [(1.0, 60, 0.0, False)], 1: [(1.0, 61, 0.0, False)], 2: [(1.0, 62, 0.0, False)], 3: [(1.0, 53, 0.0, False)]}, 62: {0: [(1.0, 61, 0.0, False)], 1: [(1.0, 62, 0.0, False)], 2: [(1.0, 63, 1.0, True)], 3: [(1.0, 54, 0.0, True)]}, 63: {0: [(1.0, 63, 0, True)], 1: [(1.0, 63, 0, True)], 2: [(1.0, 63, 0, True)], 3: [(1.0, 63, 0, True)]}}
        nS=64
        nA=4
    else:
        P= {0: {0: [(1.0, 0, 0.0, False)], 1: [(1.0, 4, 0.0, False)], 2: [(1.0, 1, 0.0, False)], 3: [(1.0, 0, 0.0, False)]}, 1: {0: [(1.0, 0, 0.0, False)], 1: [(1.0, 5, 0.0, True)], 2: [(1.0, 2, 0.0, False)], 3: [(1.0, 1, 0.0, False)]}, 2: {0: [(1.0, 1, 0.0, False)], 1: [(1.0, 6, 0.0, False)], 2: [(1.0, 3, 0.0, False)], 3: [(1.0, 2, 0.0, False)]}, 3: {0: [(1.0, 2, 0.0, False)], 1: [(1.0, 7, 0.0, True)], 2: [(1.0, 3, 0.0, False)], 3: [(1.0, 3, 0.0, False)]}, 4: {0: [(1.0, 4, 0.0, False)], 1: [(1.0, 8, 0.0, False)], 2: [(1.0, 5, 0.0, True)], 3: [(1.0, 0, 0.0, False)]}, 5: {0: [(1.0, 5, 0, True)], 1: [(1.0, 5, 0, True)], 2: [(1.0, 5, 0, True)], 3: [(1.0, 5, 0, True)]}, 6: {0: [(1.0, 5, 0.0, True)], 1: [(1.0, 10, 0.0, False)], 2: [(1.0, 7, 0.0, True)], 3: [(1.0, 2, 0.0, False)]}, 7: {0: [(1.0, 7, 0, True)], 1: [(1.0, 7, 0, True)], 2: [(1.0, 7, 0, True)], 3: [(1.0, 7, 0, True)]}, 8: {0: [(1.0, 8, 0.0, False)], 1: [(1.0, 12, 0.0, True)], 2: [(1.0, 9, 0.0, False)], 3: [(1.0, 4, 0.0, False)]}, 9: {0: [(1.0, 8, 0.0, False)], 1: [(1.0, 13, 0.0, False)], 2: [(1.0, 10, 0.0, False)], 3: [(1.0, 5, 0.0, True)]}, 10: {0: [(1.0, 9, 0.0, False)], 1: [(1.0, 14, 0.0, False)], 2: [(1.0, 11, 0.0, True)], 3: [(1.0, 6, 0.0, False)]}, 11: {0: [(1.0, 11, 0, True)], 1: [(1.0, 11, 0, True)], 2: [(1.0, 11, 0, True)], 3: [(1.0, 11, 0, True)]}, 12: {0: [(1.0, 12, 0, True)], 1: [(1.0, 12, 0, True)], 2: [(1.0, 12, 0, True)], 3: [(1.0, 12, 0, True)]}, 13: {0: [(1.0, 12, 0.0, True)], 1: [(1.0, 13, 0.0, False)], 2: [(1.0, 14, 0.0, False)], 3: [(1.0, 9, 0.0, False)]}, 14: {0: [(1.0, 13, 0.0, False)], 1: [(1.0, 14, 0.0, False)], 2: [(1.0, 15, 1.0, True)], 3: [(1.0, 10, 0.0, False)]}, 15: {0: [(1.0, 15, 0, True)], 1: [(1.0, 15, 0, True)], 2: [(1.0, 15, 0, True)], 3: [(1.0, 15, 0, True)]}}
        nS=16
        nA=4
        if is_slippery==True:
            P = {0: {0: [(0.3333333333333333, 0, 0.0, False), (0.3333333333333333, 0, 0.0, False), (0.3333333333333333, 4, 0.0, False)], 1: [(0.3333333333333333, 0, 0.0, False), (0.3333333333333333, 4, 0.0, False), (0.3333333333333333, 1, 0.0, False)], 2: [(0.3333333333333333, 4, 0.0, False), (0.3333333333333333, 1, 0.0, False), (0.3333333333333333, 0, 0.0, False)], 3: [(0.3333333333333333, 1, 0.0, False), (0.3333333333333333, 0, 0.0, False), (0.3333333333333333, 0, 0.0, False)]}, 1: {0: [(0.3333333333333333, 1, 0.0, False), (0.3333333333333333, 0, 0.0, False), (0.3333333333333333, 5, 0.0, True)], 1: [(0.3333333333333333, 0, 0.0, False), (0.3333333333333333, 5, 0.0, True), (0.3333333333333333, 2, 0.0, False)], 2: [(0.3333333333333333, 5, 0.0, True), (0.3333333333333333, 2, 0.0, False), (0.3333333333333333, 1, 0.0, False)], 3: [(0.3333333333333333, 2, 0.0, False), (0.3333333333333333, 1, 0.0, False), (0.3333333333333333, 0, 0.0, False)]}, 2: {0: [(0.3333333333333333, 2, 0.0, False), (0.3333333333333333, 1, 0.0, False), (0.3333333333333333, 6, 0.0, False)], 1: [(0.3333333333333333, 1, 0.0, False), (0.3333333333333333, 6, 0.0, False), (0.3333333333333333, 3, 0.0, False)], 2: [(0.3333333333333333, 6, 0.0, False), (0.3333333333333333, 3, 0.0, False), (0.3333333333333333, 2, 0.0, False)], 3: [(0.3333333333333333, 3, 0.0, False), (0.3333333333333333, 2, 0.0, False), (0.3333333333333333, 1, 0.0, False)]}, 3: {0: [(0.3333333333333333, 3, 0.0, False), (0.3333333333333333, 2, 0.0, False), (0.3333333333333333, 7, 0.0, True)], 1: [(0.3333333333333333, 2, 0.0, False), (0.3333333333333333, 7, 0.0, True), (0.3333333333333333, 3, 0.0, False)], 2: [(0.3333333333333333, 7, 0.0, True), (0.3333333333333333, 3, 0.0, False), (0.3333333333333333, 3, 0.0, False)], 3: [(0.3333333333333333, 3, 0.0, False), (0.3333333333333333, 3, 0.0, False), (0.3333333333333333, 2, 0.0, False)]}, 4: {0: [(0.3333333333333333, 0, 0.0, False), (0.3333333333333333, 4, 0.0, False), (0.3333333333333333, 8, 0.0, False)], 1: [(0.3333333333333333, 4, 0.0, False), (0.3333333333333333, 8, 0.0, False), (0.3333333333333333, 5, 0.0, True)], 2: [(0.3333333333333333, 8, 0.0, False), (0.3333333333333333, 5, 0.0, True), (0.3333333333333333, 0, 0.0, False)], 3: [(0.3333333333333333, 5, 0.0, True), (0.3333333333333333, 0, 0.0, False), (0.3333333333333333, 4, 0.0, False)]}, 5: {0: [(1.0, 5, 0, True)], 1: [(1.0, 5, 0, True)], 2: [(1.0, 5, 0, True)], 3: [(1.0, 5, 0, True)]}, 6: {0: [(0.3333333333333333, 2, 0.0, False), (0.3333333333333333, 5, 0.0, True), (0.3333333333333333, 10, 0.0, False)], 1: [(0.3333333333333333, 5, 0.0, True), (0.3333333333333333, 10, 0.0, False), (0.3333333333333333, 7, 0.0, True)], 2: [(0.3333333333333333, 10, 0.0, False), (0.3333333333333333, 7, 0.0, True), (0.3333333333333333, 2, 0.0, False)], 3: [(0.3333333333333333, 7, 0.0, True), (0.3333333333333333, 2, 0.0, False), (0.3333333333333333, 5, 0.0, True)]}, 7: {0: [(1.0, 7, 0, True)], 1: [(1.0, 7, 0, True)], 2: [(1.0, 7, 0, True)], 3: [(1.0, 7, 0, True)]}, 8: {0: [(0.3333333333333333, 4, 0.0, False), (0.3333333333333333, 8, 0.0, False), (0.3333333333333333, 12, 0.0, True)], 1: [(0.3333333333333333, 8, 0.0, False), (0.3333333333333333, 12, 0.0, True), (0.3333333333333333, 9, 0.0, False)], 2: [(0.3333333333333333, 12, 0.0, True), (0.3333333333333333, 9, 0.0, False), (0.3333333333333333, 4, 0.0, False)], 3: [(0.3333333333333333, 9, 0.0, False), (0.3333333333333333, 4, 0.0, False), (0.3333333333333333, 8, 0.0, False)]}, 9: {0: [(0.3333333333333333, 5, 0.0, True), (0.3333333333333333, 8, 0.0, False), (0.3333333333333333, 13, 0.0, False)], 1: [(0.3333333333333333, 8, 0.0, False), (0.3333333333333333, 13, 0.0, False), (0.3333333333333333, 10, 0.0, False)], 2: [(0.3333333333333333, 13, 0.0, False), (0.3333333333333333, 10, 0.0, False), (0.3333333333333333, 5, 0.0, True)], 3: [(0.3333333333333333, 10, 0.0, False), (0.3333333333333333, 5, 0.0, True), (0.3333333333333333, 8, 0.0, False)]}, 10: {0: [(0.3333333333333333, 6, 0.0, False), (0.3333333333333333, 9, 0.0, False), (0.3333333333333333, 14, 0.0, False)], 1: [(0.3333333333333333, 9, 0.0, False), (0.3333333333333333, 14, 0.0, False), (0.3333333333333333, 11, 0.0, True)], 2: [(0.3333333333333333, 14, 0.0, False), (0.3333333333333333, 11, 0.0, True), (0.3333333333333333, 6, 0.0, False)], 3: [(0.3333333333333333, 11, 0.0, True), (0.3333333333333333, 6, 0.0, False), (0.3333333333333333, 9, 0.0, False)]}, 11: {0: [(1.0, 11, 0, True)], 1: [(1.0, 11, 0, True)], 2: [(1.0, 11, 0, True)], 3: [(1.0, 11, 0, True)]}, 12: {0: [(1.0, 12, 0, True)], 1: [(1.0, 12, 0, True)], 2: [(1.0, 12, 0, True)], 3: [(1.0, 12, 0, True)]}, 13: {0: [(0.3333333333333333, 9, 0.0, False), (0.3333333333333333, 12, 0.0, True), (0.3333333333333333, 13, 0.0, False)], 1: [(0.3333333333333333, 12, 0.0, True), (0.3333333333333333, 13, 0.0, False), (0.3333333333333333, 14, 0.0, False)], 2: [(0.3333333333333333, 13, 0.0, False), (0.3333333333333333, 14, 0.0, False), (0.3333333333333333, 9, 0.0, False)], 3: [(0.3333333333333333, 14, 0.0, False), (0.3333333333333333, 9, 0.0, False), (0.3333333333333333, 12, 0.0, True)]}, 14: {0: [(0.3333333333333333, 10, 0.0, False), (0.3333333333333333, 13, 0.0, False), (0.3333333333333333, 14, 0.0, False)], 1: [(0.3333333333333333, 13, 0.0, False), (0.3333333333333333, 14, 0.0, False), (0.3333333333333333, 15, 1.0, True)], 2: [(0.3333333333333333, 14, 0.0, False), (0.3333333333333333, 15, 1.0, True), (0.3333333333333333, 10, 0.0, False)], 3: [(0.3333333333333333, 15, 1.0, True), (0.3333333333333333, 10, 0.0, False), (0.3333333333333333, 13, 0.0, False)]}, 15: {0: [(1.0, 15, 0, True)], 1: [(1.0, 15, 0, True)], 2: [(1.0, 15, 0, True)], 3: [(1.0, 15, 0, True)]}}

    # Make gym environment
    env = gym.make(name,is_slippery=True,render_mode='human')
    print("\n" + "-" * 25 + "\nBeginning Policy Iteration\n" + "-" * 25)

    V_pi, p_pi = policy_iteration(P, nS, nA, gamma=0.9, tol=1e-3)
    render_single(env, p_pi, 100)


    print("\n" + "-" * 25 + "\nBeginning Value Iteration\n" + "-" * 25)
    V_vi, p_vi = value_iteration(P, nS, nA, gamma=0.9, tol=1e-3)
    render_single(env, p_vi, 100)