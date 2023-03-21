# state space is (coods 1,coods 2, direction, obstacle in front, obstacle in left, obstacle in right) 4*4*3*8

import gym
import numpy as np 
import random
import matplotlib.pyplot as plt

env = gym.make('MiniGrid-Dynamic-Obstacles-6x6-v0')
# env = gym.make('MiniGrid-Dynamic-Obstacles-6x6-v0',render_mode='human')

obs=env.reset()[0]
iteration_list,epoch_list,reward_list=[],[],[]
Q_lookup={}

gamma= 0.7
alpha=1/10
Lambda=0.9
epsilon=1
max_epoch,min_epoch=2000,-2000

def update(prev_pos,new_pos,prev_action,new_action,imm_reward,Q_table,Traces):
    Q_next = Q_table[new_pos][new_action]
    td_error = (imm_reward + gamma*Q_next) - Q_table[prev_pos][prev_action]
    for states in list(Traces.keys()):
        for act in range(3):
            Traces[states][act]= gamma*Lambda*Traces[states][act]
            Q_table[states][act]=Q_table[states][act]+alpha*td_error*Traces[states][act]

def update(pos,action,mc_return,Q_table):
    error = mc_return - Q_table[pos][action]
    Q_table[pos][action] = Q_table[pos][action] + alpha * error

def epsilon_greedy_policy(Epsilon,pos,Q_table,n_epoch):
    prob_decider=random.uniform(0,1)
    if(Epsilon>=prob_decider):
        Act=random.randint(0,2)
    else:
        Act=np.argmax(list(Q_table[pos].values())) # conversion of dict.values() in list is important for correct greddy action selection.
        # if(n_epoch<30):
        #     print(n_epoch,'iteration and action chosen greedily')
    return Act

def cal_update(Episode_history,Qtable):
    for i in range(0, len(Episode_history)-1):
        lam_return=cal_lambda_return(Episode_history[i:])
        update(Episode_history[i][0],Episode_history[i][1],lam_return,Qtable)

def cal_lambda_return(Episode_history):
    sum_return=0
    gamma_p=1
    for step in Episode_history:
        sum_return= sum_return + gamma_p*step[2]
        gamma_p=gamma_p*gamma
    return sum_return


k,n=0,0
for i in range(min_epoch,int(max_epoch*1.4)):

    epoch_list.append(i+min_epoch)
    done,n,reward=0,1,0
    episode_data=[]
    if(i==2000):
        env = gym.make('MiniGrid-Dynamic-Obstacles-6x6-v0',render_mode='human')
    if(i==2050):
        env = gym.make('MiniGrid-Dynamic-Obstacles-6x6-v0')
    if(i>0 or (i<0 and i%50==0)):
        obs=env.reset()[0]

    print('--'*50,'\n',i,'epoch')

    if(i>0 and i<max_epoch+1):
        epsilon=1-i/max_epoch 

    # if(i>(min_epoch/3) and i<max_epoch*1.1 and i%20==0):
    #     print(Q_lookup)

    while(n<500 and not done):
        env.render()

        pos=(env.agent_pos[0],env.agent_pos[1],env.agent_dir,int(obs['image'][3][5][0]==6),int(obs['image'][2][6][0]==6),int(obs['image'][4][6][0]==6))
        if (not pos in Q_lookup):
            Q_lookup[pos]={0:0.0,1:0.0,2:0.0}
            k=k+1
            print(pos, 'state seen', k)

        act=epsilon_greedy_policy(epsilon,pos,Q_lookup,n)

        obs,reward,done,d,e=env.step(act)  # taking a step in the environment
        episode_data.append([pos,act,reward])

        if(reward):
            print(n,'itreartion has reward',reward)

        if(not n%100 or done):
            print("iteration",n,'and done is',done)
        n=n+1
    episode_data.append([pos,act,0])
    cal_update(episode_data,Q_lookup)
    iteration_list.append(n)
    reward_list.append(reward)

print(Q_lookup,'\n\n',epoch_list,'\n\n',reward_list,'\n\n',iteration_list)
x = np.arange(len(reward_list))
plt.plot(x,reward_list)
plt.xlabel('No. of Epochs')
plt.ylabel('No. of steps required to reach the goal')
plt.title('Q-learning applied on MiniGrid-Empty-6x6-v0')
plt.show()

