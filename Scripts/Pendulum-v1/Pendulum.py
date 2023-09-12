
import gym
import numpy as np 
import random
import torch
import math
import copy
from torch.distributions.normal import Normal
from torch.autograd import variable
import torch.nn as nn
import matplotlib.pyplot as plt
# import mujoco

env = gym.make("Pendulum-v1", g=9.81)
env.reset()
done=False
q=False
# while not done and not q:
#     #env.reset()
#     a=np.random.uniform(-2.0,2.0)
#     obs, rew, done, info,q = env.step([a])        


critic = torch.nn.Sequential(
    
        torch.nn.Linear(3,128),
        torch.nn.ReLU(),
        # torch.nn.Linear(128,128),
        # torch.nn.ReLU(),
        # torch.nn.Linear(128,64),
        # torch.nn.ReLU(),
        # torch.nn.Linear(64,32),
        # torch.nn.ReLU(),
        #torch.nn.Softmax(dim=-1),
        torch.nn.Linear(128,1)
    )
actor = torch.nn.Sequential(
    
        torch.nn.Linear(3,128),
        torch.nn.ReLU(),
        # torch.nn.Linear(128,128),
        # torch.nn.ReLU(),
        # torch.nn.Linear(128,64),
        # torch.nn.ReLU(),
        # torch.nn.Linear(64,32),
        # torch.nn.ReLU(),
        #torch.nn.Softmax(dim=-1),
        torch.nn.Linear(128,2)
    )
adv = torch.nn.Sequential(
    
        torch.nn.Linear(3,128),
        torch.nn.ReLU(),
        torch.nn.Linear(128,128),
        torch.nn.ReLU(),
        torch.nn.Linear(128,64),
        torch.nn.ReLU(),
        torch.nn.Linear(64,16),
        torch.nn.ReLU(),
        #torch.nn.Softmax(dim=-1),
        torch.nn.Linear(16,1)
    )
tacr=copy.deepcopy(critic)
taac=copy.deepcopy(actor)
def train_critic(xi,yi):
    critoptimizer= torch.optim.SGD(critic.parameters(),lr=0.000001)
    #forward
    yp=critic(xi)
    #print(yp,y)
    lo=nn.MSELoss()
    los=lo(yp,yi.detach())
    #backward
    los.backward()
    #upgrade
    critoptimizer.step()
    critoptimizer.zero_grad()
def train_actor(x,nx,rew,actiont,logp):
    # x=torch.tensor([x[0],x[1],x[2]], dtype=torch.float32)
    # nx=torch.tensor([nx[0],nx[1],nx[2]], dtype=torch.float32)
    actoptimizer= torch.optim.SGD(actor.parameters(),lr=0.001)
    with torch.no_grad():
        Vs=tacr(x)
        Vsn=tacr(nx)
        #praqrut kuteja aditya shirwadkar
    adv = rew + 0.99*Vsn - Vs
    para = actor(x)
    #print(para)
    std=para[1]
    #print(i)
    std = torch.clamp(para[1],-20,2)
    std=torch.exp(std)
    #print(para[0],std)
    dist = Normal(para[0],std)
   # print(actiont)
    #sample = dist.sample()
    #print(sample)
    actor_loss = -logp*adv#dist.log_prob(actiont)*adv
    actor_loss.mean().backward()
    actoptimizer.step()
    actoptimizer.zero_grad()
    return actor_loss.detach().numpy() , std
    
def tsav(X):
    with torch.no_grad():
        x=torch.tensor([X[0],X[1],X[2]], dtype=torch.float32)
        aa=tacr(x)
        savv=(aa).detach().numpy()
    return savv 
def sav(X,action): 
    with torch.no_grad():
        x=torch.tensor([X[0],X[1],X[2]], dtype=torch.float32)
        aa=critic(x)
        savv=(aa).detach().numpy()
    return savv 
# def difference(X):
#     aa=sav(X,0)
#     bb=tsav(X,0)
#     return aa-bb

def policy(x):
    #with torch.no_grad():
        acta=np.zeros(1)
        x=torch.from_numpy(x)
        para = actor(x)
        std = torch.clamp(para[1],-20,2)
        std=torch.exp(std)
        print(para)
        dist = Normal(para[0],std)
        act=dist.sample()
        #print(act)
        y_t = torch.tanh(act)
        action = 2 * y_t 
        logp=dist.log_prob(action)
        #print(action)
        #print('acttttttttttttttttttttttttttttttt',act)
        act=action.detach()
        acta[0]=act
        #print(act.dtype)
        return acta,logp
#ac=policy([0.1,0.2,0.4])
#############################################################################################################
pobsm=[]
obsm=[]
actionm=[]
rewm=[]
rewlist=[]
rewlist2=[]
donem=[]
lossm=[]
epsilonl=[]
stdll=[]
obs=np.zeros(4)
rplymsize=10000
tt=2000
kk=0
ed=1/(tt-1)
count=0
env.reset()
done=False
trunc=False
while len(obsm)<rplymsize:
    obs=env.reset()
    obs=obs[0]
    done=False
    trunc=False
    reward=0
    #time=0
    r=0
    while  not(done) and not (trunc):
        action,logp=policy(obs)
        #print(action)
        pobs=np.copy(obs)
        obs, rew, done, trunc,q = env.step(action) 
        pobsm.append(pobs)
        obsm.append(obs)
        actionm.append(action)
        rewm.append(rew)
        donem.append(done)
        if len(obsm)>rplymsize:
            pobsm.pop(0)
            obsm.pop(0)
            actionm.pop(0)
            rewm.pop(0)
            donem.pop(0)
            break
#     if done==True or trunc==True:
#         env.reset()
#         done=False
#         trunc=False
    # action=random.randint(0,1)
    # pobs=np.copy(obs)
    # #obs, rew, done, trunc,q = env.step(action) 
    # pobsm.append(pobs)
    # obsm.append(obs)
    # actionm.append(action)
    # rewm.append(rew)
    # donem.append(done)
    #print("fill")
#env = gym.make("CartPole-v1",render_mode='human')

for i in range (0,tt):
    if i%50==0:
        print(i)
    obs=env.reset()
    obs=obs[0]
    done=False
    trunc=False
    reward=0
    r=0
    while  not(done) and not (trunc):
        action,logp=policy(obs)
        #print(logp)
        pobs=np.copy(obs)
        obs, rew, done, trunc,q = env.step(action) 
        r+=rew
        Y=((1-done)*0.9*tsav(obs)+rew)
        pt=torch.tensor(pobs, dtype=torch.float32)
        pnt=torch.tensor(obs, dtype=torch.float32)
        yt=torch.tensor(Y, dtype=torch.float32)
        rewt=torch.tensor(rew,dtype=torch.float32)
        actiont=torch.tensor(action,dtype=torch.float32)
        train_critic(pt,yt)
        losss,stdv=train_actor(pt,pnt,rewt,actiont,logp)
        actiont=torch.tensor([action],dtype=torch.float32)
        pobsm.append(pobs)
        obsm.append(obs)
        actionm.append(action)
        rewm.append(rew)
        donem.append(done)
        if len(obsm)>rplymsize:
            pobsm.pop(0)
            obsm.pop(0)
            actionm.pop(0)
            rewm.pop(0)
            donem.pop(0)
    if i<tt/5:
        k=128
    else:
        k=128
    if len(obsm)>=rplymsize:
        #for pp in range (0,2):
            sample=random.sample(range(0,len(obsm)), int(k))
            #random.shuffle(sample)
            losss=0
            for tr in range(0,1):
                ya=[]#np.zeros((32,1))
                pobsl=[]#np.zeros((32,4))
                obsl=[]
                rewl=[]
                actl=[]
                for itm in sample:
                    #train_actor(pobsm[itm],obsm[itm],rewm[itm])
                    with torch.no_grad():
                        Y=((1-donem[itm])*0.9*tsav(obsm[itm])+rewm[itm])#*0.01+0.99*tsav(pobsm[itm],actionm[itm])
                        #Y=torch.tensor([Y],dtype=torch.float32)
                        #print(Y)
                        ya.append([Y])
                        pobsa=np.zeros(3)
                        obsa=np.zeros(3)
                        rewa=np.zeros(1)
                        for arc in range(0,3):
                            pobsa[arc]=((pobsm[itm][arc]))#[ai][arc]=torch.tensor(pobsm[itm][arc+1], dtype=torch.float32)
                            obsa[arc]=(obsm[itm][arc])
                        #rewa=rewm[itm]
                        #pobsa[]=((actionm[itm]))#[ai][3]=torch.tensor(actionm[itm], dtype=torch.float32)
                        pobsl.append(pobsa)#torch.tensor([pobsa],dtype=torch.float32))
                        obsl.append(obsa)
                        rewl.append([rewm[itm]])
                        actl.append([actionm[itm]])
                pt=torch.tensor(np.array(pobsl), dtype=torch.float32)
                pnt=torch.tensor(np.array(obsl), dtype=torch.float32)
                yt=torch.tensor(np.array(ya), dtype=torch.float32)
                rewt=torch.tensor(np.array(rewa),dtype=torch.float32)
                actiont=torch.tensor(np.array(actl),dtype=torch.float32)
                #losss+=train_critic(pobz,actionm[itm],Y)    
                train_critic(pt,yt)
                losss,stdv=train_actor(pt,pnt,rewt,actiont)
                #print(losss)
                lossm.append(losss)
                stdll.append(stdv)
                    #print(losss)
            #if k%(k/2)==0:
            with torch.no_grad():
                tacr=copy.deepcopy(critic)      
                    # print(difference(obs))
    #print("kkkkkkkkkkkkk")
   
    #print(difference(obs))
    # tau = 0.001  # Target network update rate
    # #Update target network weights using soft updating
    # for target_param, critic_param in zip(tarmod.parameters(), critic.parameters()):
    #     target_param.data.copy_(tau * critic_param.data + (1 - tau) * target_param.data)

    #print(difference(obs))
        ####
    #print("reward",r) 
    rewlist.append(r)

    r=0
    obs=env.reset()
    obs=obs[0]
    done=False
    trunc=False
    while not (done) and not( trunc):
        with torch.nograd():
            action,logp=policy(obs)
            obs, rew, done, trunc,q = env.step(action) 
            r+=rew
    obs=env.reset()
    obs=obs[0]
    done=False
    trunc=False

    rewlist2.append(r)

plt.plot(rewlist)
plt.show()
plt.plot(rewlist2)
plt.show()
env = gym.make("Pendulum-v1",g=9.81,render_mode='human')
for i in range(0,10):
    done=False
    env.reset()
    while not done:
        action=policy(obs)
        obs, rew, done, info,q = env.step(action) 
        