import gym
import numpy as np 
import random
import matplotlib.pyplot as plt

env = gym.make('MiniGrid-Empty-6x6-v0')
# env = gym.make('MiniGrid-Empty-6x6-v0',render_mode='human')

env.reset()
iteration_list,epoch_list,reward_list=[],[],[]
Q_lookup={}  # n * n * directions * actions

gamma= 0.7 # discount factor
alpha=1/10  # step size can be reduced further
Lambda=0.9
epsilon=1

max_epoch,min_epoch=80,0

def update(pos,action,lambda_return,Q_table):
    td_error = lambda_return - Q_table[pos][action]
    Q_table[pos][action] = Q_table[pos][action] + alpha * td_error

def epsilon_greedy_policy(Epsilon,pos,Q_table):
    prob_decider=random.uniform(0,1)
    if(Epsilon>=prob_decider):
        Act=random.randint(0,2)
    else:
        Act=np.argmax(list(Q_table[pos].values())) # conversion of dict.values() in list is important for correct greddy action selection.
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
    env.reset()
    print('--'*60,'\n',i,'epoch')
    if(i>0 and i<max_epoch+1):
        epsilon=1-i/max_epoch

    while(n<700 and not done):
        env.render()
            
        pos=(env.agent_pos[0],env.agent_pos[1],env.agent_dir)
        if (not pos in Q_lookup):
            Q_lookup[pos]={0:0.0,1:0.0,2:0.0}

        act=epsilon_greedy_policy(epsilon,pos,Q_lookup)
        a,reward,done,d,e=env.step(act)
        episode.append([pos,act,reward])
        if(reward<0): # eliminating -ve rewards as it might get propagated in wrong way in initial steps only.
            reward=0
        if(reward):
            print(n,'itreartion has reward',reward)
        if(not n%200 or done):
            print("iteration",n,'and done is',done)
        n=n+1

    episode.append([pos,act,0])
    cal_update(episode,Q_lookup)
    iteration_list.append(n)
    reward_list.append(reward)

print(Q_lookup,'\n\n',epoch_list,'\n\n',reward_list,'\n\n',iteration_list)
plt.plot(reward_list)
plt.xlabel('No. of Epochs')
plt.ylabel('Reward Function')
plt.title('SARSA-Lambda applied on MiniGrid-Empty-6x6-v0')
plt.show()