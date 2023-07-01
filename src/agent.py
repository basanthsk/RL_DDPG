import torch 
import numpy as np
import torch.nn as nn 
import torch.optim as optim 
import torch.nn.functional as F
from src.networks import ActorNetwork,CriticNetwork
from src.state_representation import StateRepresentation
from src.noise import OUActionNoise
from src.buffer import ReplayBuffer



class Agent:
    def __init__(self,alpha,beta,tau,input_dims,n_actions,users_num,games_num,fc1_dims =400,fc2_dims=300,
                 gamma= 0.99, max_size = 100000,batch_size =64,state_size=5):
        super(Agent,self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta
        self.tau = tau
        self.batch_size = batch_size
        
        self.memory = ReplayBuffer(max_size,input_dims,n_actions)
        self.noise = OUActionNoise(mu = np.zeros(n_actions))
        self.actor = ActorNetwork(alpha,input_dims,n_actions,fc1_dims,fc2_dims,name='actor')
        self.critic = CriticNetwork(beta,input_dims,n_actions,fc1_dims,fc2_dims,name='critic')
        self.target_actor = ActorNetwork(alpha,input_dims,n_actions,fc1_dims,fc2_dims, name = 'target_actor')
        self.target_critic = CriticNetwork(beta,input_dims,n_actions,fc1_dims,fc2_dims,name="target_critic")
        self.q  = StateRepresentation(users_num, games_num,state_size=state_size)
        self.update_network_parameters(tau=1)
        
    def choose_action(self,observation):
        self.actor.eval()
        state = observation.to(self.actor.device) 
        mu_prime = self.actor.forward(state).to(self.actor.device)
        # mu_prime = mu + torch.tensor(self.noise(),dtype = torch.float).to(self.actor.device) 
        
        self.actor.train()
        
        return mu_prime.cpu().detach().numpy()[0]      
    
    def remember(self,state,action,reward,state_,done):
        self.memory.store_transition(state,action,reward,state_,done)
        
    def save_models(self):
        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.critic.save_checkpoint()
        self.target_critic.save_checkpoint()
    
    def load_models(self):
        self.actor.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.critic.load_checkpoint()
        self.target_critic.load_checkpoint()
        
    def learn(self):
        
        if self.memory.mem_cntr < self.batch_size:
            return
        states,actions,rewards,states_,done = \
            self.memory.sample_buffer(self.batch_size)  
        states = torch.tensor(states,dtype=torch.float).to(self.actor.device)
        actions = torch.tensor(actions,dtype=torch.float).to(self.actor.device)
        states_ = torch.tensor(states_,dtype=torch.float).to(self.actor.device)
        rewards = torch.tensor(rewards,dtype=torch.float).to(self.actor.device)
        done= torch.tensor(done).to(self.actor.device)
        
        
        target_actions = self.target_actor.forward(states_)  
        critic_value_ = self.target_critic.forward(states_,target_actions)
        critic_value = self.critic.forward(states,actions)  
        
        critic_value_[done] = 0.0
        critic_value_ = critic_value_.view(-1)
        
        #Bellaman Equation
        target = rewards + self.gamma*critic_value_
        target = target.view(self.batch_size,1)
 
        self.q.optimizer.zero_grad()
        self.q.optimizer.step()
         
        self.critic.optimizer.zero_grad()
        critic_loss = F.mse_loss(target,critic_value)
        critic_loss.backward()
        self.critic.optimizer.step()
        
        self.actor.optimizer.zero_grad()
        actor_loss = -self.critic.forward(states,self.actor.forward(states))
        actor_loss = torch.mean(actor_loss)
        actor_loss.backward()
        self.actor.optimizer.step()
        
        self.update_network_parameters()
        
        
        return critic_loss.cpu().detach().numpy()
        
    def update_network_parameters(self,tau=None):
        if tau is None:
            tau = self.tau        
        
        actor_params = self.actor.named_parameters()
        critic_params = self.critic.named_parameters()
        target_actor_params = self.target_actor.named_parameters()
        target_critic_params = self.target_critic.named_parameters()
        
        critic_state_dict = dict(critic_params)
        actor_state_dict = dict(actor_params)
        target_actor_state_dict = dict(target_actor_params)
        target_critic_state_dict = dict(target_critic_params)
        
        for name in critic_state_dict:
            critic_state_dict[name] = tau*critic_state_dict[name].clone()+ \
                                    (1-tau)*target_critic_state_dict[name].clone()
        
        for name in actor_state_dict:
            actor_state_dict[name] = tau*actor_state_dict[name].clone()+ \
                                    (1-tau)*target_actor_state_dict[name].clone()   
                                    
        self.target_critic.load_state_dict(critic_state_dict)
        self.target_actor.load_state_dict(actor_state_dict)                                                         
            
        