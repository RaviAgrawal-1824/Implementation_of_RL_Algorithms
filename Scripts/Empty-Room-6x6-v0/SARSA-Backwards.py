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

def update(prev_pos,new_pos,prev_action,new_action,imm_reward,Q_table,Traces):
    Q_next = Q_table[new_pos][new_action]
    td_error = (imm_reward + gamma*Q_next) - Q_table[prev_pos][prev_action]
    for states in list(Traces.keys()):
        for act in range(3):
            Traces[states][act]= gamma*Lambda*Traces[states][act]
            Q_table[states][act]=Q_table[states][act]+alpha*td_error*Traces[states][act]

def epsilon_greedy_policy(Epsilon,pos,Q_table):
    prob_decider=random.uniform(0,1)
    if(Epsilon>=prob_decider):
        Act=random.randint(0,2)
    else:
        Act=np.argmax(list(Q_table[pos].values())) # conversion of dict.values() in list is important for correct greddy action selection.
    return Act

k,n=0,0
for i in range(min_epoch,int(max_epoch*1.4)):

    epoch_list.append(i+min_epoch)
    done,n,reward=0,1,0
    Eligibility_traces={}
    env.reset()
    print('--'*60,'\n',i,'epoch')
    if(i>0 and i<max_epoch+1):
        epsilon=1-i/max_epoch

    while(n<700 and not done):
        env.render()
            
        prev_pos=(env.agent_pos[0],env.agent_pos[1],env.agent_dir)
        if (not prev_pos in Q_lookup):
            Q_lookup[prev_pos]={0:0.0,1:0.0,2:0.0}

        if (not prev_pos in Eligibility_traces):
            Eligibility_traces[prev_pos]={0:0.0,1:0.0,2:0.0}

        try: # implemented so that if new_act is not initialised yet then its value is obtained through epsilon-greedy policy else directly equated to new act. 
            act=new_act
        except:
            act=epsilon_greedy_policy(epsilon,prev_pos,Q_lookup,n)

        a,reward,done,d,e=env.step(act)
        Eligibility_traces[prev_pos][act]=Eligibility_traces[prev_pos][act] + 1
        if(reward<0):
            reward=0
        if(reward):
            print(n,'itreartion has reward',reward)

        new_pos=(env.agent_pos[0],env.agent_pos[1],env.agent_dir)
        if (not new_pos in Q_lookup):
            Q_lookup[new_pos]={0:0.0,1:0.0,2:0.0}
        if (not new_pos in Eligibility_traces):
            Eligibility_traces[new_pos]={0:0.0,1:0.0,2:0.0}

        new_act=epsilon_greedy_policy(epsilon,new_pos,Q_lookup)
        update(prev_pos,new_pos,act,new_act,reward,Q_lookup,Eligibility_traces)

        if(not n%200 or done):
            print("iteration",n,'and done is',done)
        n=n+1
    print(Eligibility_traces)
    iteration_list.append(n)
    reward_list.append(reward)

print(Q_lookup,'\n\n',epoch_list,'\n\n',reward_list,'\n\n',iteration_list)
plt.plot(reward_list)
plt.xlabel('No. of Epochs')
plt.ylabel('Reward Function')
plt.title('SARSA-Backwards applied on MiniGrid-Empty-6x6-v0')
plt.show()