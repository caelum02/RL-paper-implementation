from model import ICM, A2C_Model
from torch.nn.functional import one_hot
import tool
import torch.optim as optim
import torch

class ICM_A2C_agent():
    def __init__(self, action_dim, beta=0.2, lamb=0.1, gamma=0.99):
        self.a2c = A2C_Model(action_dim)
        self.icm = ICM(action_dim)
        
        self.optimizer_icm = optim.Adam(self.a2c.parameters(), lr=0.0001)
        self.optimizer_a2c = optim.Adam(self.icm.parameters(), lr=0.0001)
        
        self.gamma = gamma
        self.beta = beta 
        self.lamb = lamb 

    def train(self, buffers):
        buff = buffers.as_tensor()

        curiosity, act_dist_pred = self.icm(buff['obs'], buff['act'], buff['nxt_obs'])

        buff['rwd'] += curiosity.detach()

        self.optimizer_icm.zero_grad()
        self.optimizer_a2c.zero_grad()


        loss_F = curiosity.mean()

        loss_I = - (torch.log(act_dist_pred.probs)*one_hot(buff['act'].long(), num_classes=12)).mean()

        discounted_rets = tool.discount(buff['rwd'], self.gamma)

        act_dist, critic = self.a2c(buff['obs']) 
        adv = discounted_rets - critic
        
        logp = act_dist.log_prob(buff['act'])
        loss_actor = -(logp * adv.detach()).mean()
        loss_critic = 0.5 * adv.pow(2).mean()


        loss_a2c = self.lamb * (loss_actor + loss_critic - 0.02 * act_dist.entropy().mean())
        loss_a2c.backward()
        
        loss_icm = self.beta * loss_F + (1-self.beta) * loss_I
        loss_icm.backward()

        self.optimizer_icm.step()
        self.optimizer_a2c.step()

        return loss_F, loss_I, loss_a2c 
    
    def get_action(self, obs):
        policy, _ = self.a2c.forward(obs)
        act = policy.sample().item()

        return act

    def save(self, path='./model'):
        torch.save(self.a2c.state_dict(), path+'/a2c.pt')
        torch.save(self.icm.state_dict(), path+'/icm.pt')

    def load(self, path='./model'):
        self.a2c.load_state_dict(torch.load(path+'/a2c.pt'))
        self.icm.load_state_dict(torch.load(path+'/icm.pt'))