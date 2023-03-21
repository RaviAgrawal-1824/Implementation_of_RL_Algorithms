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

def update(pos,action,lambda_return,Q_table):
    td_error = lambda_return - Q_table[pos][action]
    Q_table[pos][action] = Q_table[pos][action] + alpha * td_error

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
    for i in range(0, len(Episode_history)):
        lambda_return=cal_lambda_return(Episode_history[i:],Qtable)
        update(Episode_history[i][0],Episode_history[i][1],lambda_return,Qtable)

def cal_lambda_return(Episode_history,Qtable):
    sum_return,final_return=0,0
    product,gamma_p=(1-Lambda),1
    for i in range(0, len(Episode_history)-1): # last one handled differently
        sum_return= sum_return + gamma_p*(Episode_history[i][2])
        gamma_p=gamma_p*gamma
        final_return=final_return+product*(sum_return + gamma_p*Qtable[Episode_history[i+1][0]][Episode_history[i+1][1]])
        product=product*Lambda
    final_return=final_return+ (Lambda**(len(Episode_history)-1))*(gamma_p*(sum_return))
    return final_return

k,n=0,0
for i in range(min_epoch,int(max_epoch*1.4)):

    epoch_list.append(i+min_epoch)
    done,n,reward=0,1,0
    episode=[]
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
        episode.append([pos,act,reward])

        if(reward):
            print(n,'itreartion has reward',reward)
        if(not n%100 or done):
            print("iteration",n,'and done is',done)
        n=n+1
    episode.append([pos,act,0])
    cal_update(episode,Q_lookup)
    iteration_list.append(n)
    reward_list.append(reward)

print(Q_lookup,'\n\n',epoch_list,'\n\n',reward_list,'\n\n',iteration_list)
x = np.arange(len(reward_list))
plt.plot(x,reward_list)
plt.xlabel('No. of Epochs')
plt.ylabel('Reward Function')
plt.title('SARSA-Lambda implemented on MiniGrid-Dynamic-Obstacles-6x6-v0')
plt.show()

