from utils import *
# Agent Base
class Agent:
    def __init__(self, policy, opt, actions, device, gamma=1):
        self.name = "Base"
        self.policy = policy
        self.opt = opt
        self.gamma=gamma
        self.actions = actions
        self.device=device
    
    def load(self, data):
        self.policy.load_state_dict(data['policy'])
    
    def save(self):
        return {'policy': self.policy.state_dict()}
    
    def prep(self, s): # prepare obs state to device
        return torch.from_numpy(s).float().view(1, -1).to(self.device)

    def forward(self, s): 
        with torch.no_grad():
            return self.policy(self.prep(s)).max(1)[1][0]
    
    def best(self, s): # return best actions
        return self.actions[self.forward(s)]

    def train(self, s, a, r, DONE):
        pass


    
class Policy:
    def __init__(self, env) -> None:
        self.env = env
        self.action_space = env.action_space
        self.action_list = list(range(env.action_space.start, env.action_space.start + env.action_space.n)) # for discrete action space
        
    # For envirionment emulation on CPU
    def prob(self, a, s):       # for single S in np.array form
        return

    def rand(self):             # only for single time
        return self.action_space.sample()
    
    def greedy(self,s):
        return
    
    def eps_greedy(self,s):
        return

    # For training: work on GPU in torch.Tensor format
    def prob_torch(self, a, s): # for multiple S in torch.tensor form
        return


    


