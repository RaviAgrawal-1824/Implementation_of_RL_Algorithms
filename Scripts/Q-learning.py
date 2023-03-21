import gym
import numpy as np 
import random
import matplotlib.pyplot as plt

env = gym.make('MiniGrid-Empty-8x8-v0')
# env = gym.make('MiniGrid-Empty-Random-6x6-v0',render_mode='human')
# env = gym.make('MiniGrid-Empty-6x6-v0',render_mode='human')
# env = gym.make('MiniGrid-Empty-8x8-v0',render_mode='human')
env.reset()
iteration_list,epoch_list,reward_list=[],[],[]
Q_lookup={}  # n * n * directions * actions

gamma= 0.7
alpha=1/10  # step size can be reduced further
Lambda=0.9
epsilon=1

max_epoch,min_epoch=80,0

def update(prev_pos,new_pos,action,imm_reward,Q_table):
    Qval_max = max(list(Q_table[new_pos].values()))
    change = (imm_reward+gamma*Qval_max) - Q_table[prev_pos][action]
    Q_table[prev_pos][action]=Q_table[prev_pos][action]+alpha*change

k,n=0,0
for i in range(min_epoch,int(max_epoch*1.4)):

    epoch_list.append(i+min_epoch)
    done,n,reward=0,1,0
    env.reset()
    print('--'*50,'\n',i,'epoch')

    if(i>0 and i<max_epoch+1):
        epsilon=1-i/max_epoch

    if(i>(min_epoch/3) and i<max_epoch*1.1 and i%7==0):
        print(Q_lookup)

    while(n<700 and not done):
        env.render()

        prev_pos=(env.agent_pos[0],env.agent_pos[1],env.agent_dir)
        if (not prev_pos in Q_lookup):
            Q_lookup[prev_pos]={0:0.0,1:0.0,2:0.0}

        prob_decider=random.uniform(0,1)  # mechanism for choosing action greedily or randomly controlled by epsilon.
        if(epsilon>=prob_decider):
            act=random.randint(0,2)
        else:
            act=np.argmax(list(Q_lookup[prev_pos].values())) # conversion of dict.values() in list is important for correct greddy action selection.
            if(n<30):
                print(n,'iteration and action chosen greedily')

        a,reward,done,d,e=env.step(act)  # taking a step in the environment
        if(reward<0):
            reward=0
        if(reward):
            print(n,'itreartion has reward',reward)

        new_pos=(env.agent_pos[0],env.agent_pos[1],env.agent_dir)
        if (not new_pos in Q_lookup):
            Q_lookup[new_pos]={0:0.0,1:0.0,2:0.0}

        update(prev_pos,new_pos,act,reward,Q_lookup)

        if(not n%200 or done):
            print("iteration",n,'and done is',done)
        n=n+1
    iteration_list.append(n)
    reward_list.append(reward)

print(Q_lookup,'\n\n',epoch_list,'\n\n',reward_list,'\n\n',iteration_list)
x = np.arange(len(reward_list))
plt.plot(x,reward_list)
plt.xlabel('No. of Epochs')
plt.ylabel('No. of steps required to reach the goal')
plt.title('Q-learning applied on MiniGrid-Empty-6x6-v0')
plt.show()

