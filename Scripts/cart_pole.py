import gym
import numpy as np 
import random
import torch
import sys
import copy
from torch.autograd import variable
import torch.nn as nn
import matplotlib.pyplot as plt
env = gym.make('CartPole-v1')


# All updates in it then copied in static model for stability.
dynamic_model = torch.nn.Sequential(
        torch.nn.Linear(5,64),
        torch.nn.ReLU(),
        torch.nn.Linear(64,128),
        torch.nn.ReLU(),
        torch.nn.Linear(128,64),
        torch.nn.ReLU(),
        torch.nn.Linear(64,1),
    )

# static model, no updates on it, return calculated through it.
static_model = copy.deepcopy(dynamic_model) 

# trains on training data and training label and returns MSE-loss as numpy
def train_dynamic_model(train_data,train_label): # train_data is observation and train_label is return calculated by us
    optimizer= torch.optim.Adam(dynamic_model.parameters(),lr=0.001)
    #forward
    train_output=dynamic_model(train_data) # return predcited by model
    loss_function=nn.MSELoss()
    train_loss=loss_function(train_output,train_label)
    #backward
    train_loss.backward()  #updating parameters
    optimizer.step()
    optimizer.zero_grad() # resetting gradients of the model 
    return train_loss.detach().numpy() # returning as numpy from tensor

def static_model_output(obs,action): # returns 1 output of static model in numpy form?
    data=torch.tensor(list(obs)+[action], dtype=torch.float32)
    with torch.no_grad():
        aa=static_model(data)
    return aa.detach().numpy()

def dynamic_model_output(obs,action): # returns 1 output of dynamic model in numpy form?
    data=torch.tensor(list(obs)+[action], dtype=torch.float32)
    with torch.no_grad():
        aa=dynamic_model(data)
    return aa.detach().numpy()

def static_dynamic_model_difference(obs,action): # returns difference between output of both models
    aa=static_model_output(obs,action)
    bb=dynamic_model_output(obs,action)
    return aa-bb

def policy(obs): # returns optimal action comapring both action values, random action if both are equal
    with torch.no_grad():
        Q_val0 = static_model_output(obs,0)
        Q_val1 = static_model_output(obs,1)
        act = np.argmax((Q_val0,Q_val1))
        if Q_val0 == Q_val1: # if not learnt or both equal then take random action and not a specific action
            act=random.randint(0,1)
    return act

#############################################################################################################

replay_memory_state,replay_memory_label=[],[] # (obs,action,reward,done,prev_obs)
loss_plot,epsilon_plot,reward_plot=[],[],[]
# previous_state_replay=[]
optimal_reward_plot=[]

replay_memory_size=5000 # replay memory size
total_epochs=1500 # number of epochs
batch_size=1024 # batch size
epsilon = 1
gamma=0.99

env.reset() # better obs=
done,truncation = False, False

# filling replay memory, not much required
while len(replay_memory_state) < replay_memory_size:
    
    obs=list(env.reset()[0])
    done, truncation = False, False

    while (not done) and (not truncation):
        action=random.randint(0,1)
        prev_obs=np.copy(obs) # present observation
        obs, reward, done, truncation,q = env.step(action)
        # previous_state_replay.append(prev_obs)
        replay_memory_state.append([obs,action,reward,done,prev_obs])
        if len(replay_memory_state) == replay_memory_size:
            break

print('replay done')

for i in range (0,total_epochs):
    
    epsilon = 1 - i/total_epochs 
    epsilon_plot.append(epsilon)
    if i%50==0:
        print(i)
    
    obs=env.reset()[0]
    done, truncation = False, False
    reward=0
    loss=0
    # r=0
    while  (not done) and (not truncation):

        if epsilon>random.uniform(0,1):
            action=random.randint(0,1)
        else:
            action=policy(obs)
        
        prev_obs = np.copy(obs) # storing present observation
        obs, rew, done, truncation,q = env.step(action) 
        
        reward += rew

        # previous_state_replay.append(prev_obs)
        replay_memory_state.append([obs,action,rew,done,prev_obs])

        while len(replay_memory_state) > replay_memory_size:
            replay_memory_state.pop(0)
            # previous_state_replay.pop(0)

    reward_plot.append(reward)
    if len(replay_memory_state) != replay_memory_size:
        print('replay buffer does not same')
        sys.exit()
    else:

        loss=0
        sample=random.sample(range(0,replay_memory_size), batch_size)
        label_list=[]  
        data_list=[] 
        # ai=0

        for item in sample: 
            #return 0 at terminal state as no further so 1 - done, max of next states are taken
            # state action value or retuen of cureent obs after step
            label=(replay_memory_state[item][2] + (1-replay_memory_state[item][3]) * gamma * np.max([static_model_output(replay_memory_state[item][4],0),static_model_output(replay_memory_state[item][4],1)]))
            label_list.append(label)
            data=list(replay_memory_state[item][0])+[replay_memory_state[item][1]]
            data_list.append(data)
        train_data=torch.tensor(data_list, dtype=torch.float32)
        train_label=torch.tensor(label_list, dtype=torch.float32)   
        loss=train_dynamic_model(train_data,train_label)
        # loss=loss+train_dynamic_model(train_data,train_label)
    loss_plot.append(loss)

    # if i%3==0:
    #     static_model=copy.deepcopy(dynamic_model)      
    static_model=copy.deepcopy(dynamic_model)      

# checking for parameters if it acts optimal here
    reward=0
    obs=list(env.reset()[0])
    done, truncation = False, False

    while (not done) and (not truncation):
            action=policy(obs)
            obs, rew, done, truncation,q = env.step(action) 
            reward += rew
    optimal_reward_plot.append(reward)
    

plt.title('DQN implemented on CartPole-v1')
plt.xlabel('No. of Epochs')
plt.ylabel('Loss function')
plt.plot(loss_plot)
plt.show()
plt.title('DQN implemented on CartPole-v1')
plt.xlabel('No. of Epochs')
plt.ylabel('Reward function with behaviour policy')
plt.plot(reward_plot)
plt.show()
plt.title('DQN implemented on CartPole-v1')
plt.xlabel('No. of Epochs')
plt.ylabel('Reward function with target policy')
plt.plot(optimal_reward_plot)
plt.show()

env = gym.make("CartPole-v1",render_mode='human')
for i in range(0,10):
    done, truncation = False, False
    obs=env.reset()[0]
    while not done:
        action=policy(obs)
        obs, reward, done, truncation,q = env.step(action) 
        