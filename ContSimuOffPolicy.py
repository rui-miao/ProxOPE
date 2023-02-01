#%%
import os
import sys
import csv
#%%
import numpy as np
import torch
from envs import ContinuousEnv
from agents import Policy
from utils import *
from collections import OrderedDict
from prox_fqe import fit_qpi_cv, fit_v0
from scipy.special import expit as scipy_expit
from torch.special import expit as torch_expit
from rkhs_torch import _to_tensor

#%%
def main():
    episode_len    = int(sys.argv[1]) # e.g., 1,2,3,4,5
    samp_size      = int(sys.argv[2]) # e.g., 256, 512, 1024,...
    epsilon        = float(sys.argv[3]) # 0.4,0.2,0.1
    device         = sys.argv[4]      # 'cuda:0', 'cuda:1',... 
    test_samp_size = int(sys.argv[5]) # e.g. 5000, 10000, 50000
    file_path      = sys.argv[6]

    if not os.path.exists(file_path):
        with open(file_path, 'a+') as csv_file:
            csv_file.write("episode_len,samp_size,epsilon,abs_error\n")
            csv_file.flush()
            os.fsync(csv_file.fileno())

    # %% Setup Continuous State Environment  t_u -> kappa_a -> mu_a
    continuousParams = {'episode_len':episode_len, 'offline':False, 
                        'alpha_0':0, 'alpha_a':0.5, 'alpha_s':[0.5, 0.5], 
                        'mu_0':0,    'mu_a':-0.25,  'mu_s':[0.5, 0.5], 
                        'kappa_0':0, 'kappa_a':-0.5, 'kappa_s':[0.5,0.5], 
                        't_0':0,      't_u':1,     't_s':[-0.5,-0.5]}

    ContEnv= ContinuousEnv(continuousParams)

    #%% Set up eps_greedy Policy
    class ContPolicy(Policy):
        def __init__(self, env, eps, device):
            super().__init__(env=env)
            self.device = device
            self.eps = eps
            with torch.no_grad():
                self.kappa_0    = torch.tensor(self.env.kappa_0)
                self.kappa_a    = torch.tensor(self.env.kappa_a)
                self.kappa_s    = torch.tensor(self.env.kappa_s)
                self.t0tukappa0 = torch.tensor(self.env.t_0 + self.env.t_u*self.env.kappa_0)
                self.tstukappas = torch.tensor(self.env.t_s + self.env.t_u*self.env.kappa_s)

        @torch.no_grad()
        def _to_device(self) -> None:
            self.kappa_0    = _to_tensor(self.kappa_0,    self.device)
            self.kappa_a    = _to_tensor(self.kappa_a,    self.device)
            self.kappa_s    = _to_tensor(self.kappa_s,    self.device)
            self.t0tukappa0 = _to_tensor(self.t0tukappa0, self.device)
            self.tstukappas = _to_tensor(self.tstukappas, self.device)

        def _to_cpu(self) -> None:
            self.kappa_0    = self.kappa_0.to('cpu').numpy()    if type(self.kappa_0) is not np.ndarray else self.kappa_0
            self.kappa_a    = self.kappa_a.to('cpu').numpy()    if type(self.kappa_a) is not np.ndarray else self.kappa_a
            self.kappa_s    = self.kappa_s.to('cpu').numpy()    if type(self.kappa_s) is not np.ndarray else self.kappa_s
            self.t0tukappa0 = self.t0tukappa0.to('cpu').numpy() if type(self.t0tukappa0) is not np.ndarray else self.t0tukappa0
            self.tstukappas = self.tstukappas.to('cpu').numpy() if type(self.tstukappas) is not np.ndarray else self.tstukappas

        @torch.no_grad()
        def prob_torch(self, a, s):
            ind = (self.kappa_0 + s@self.kappa_s \
                 + 1-2*self.kappa_a*torch_expit(self.t0tukappa0 + s@self.tstukappas) \
                      + s[:,0] - 2*s[:,1] > 0).type(torch.float64)
            if a==1:
                return torch.abs(ind - self.eps).view(-1,1)
            else:
                return (1. - torch.abs(ind - self.eps)).view(-1,1)

        def eps_greedy(self, obs):
            s = obs['S']
            ind = int(self.kappa_0 + s@self.kappa_s \
                + 1-2*self.kappa_a*scipy_expit(self.t0tukappa0 + s@self.tstukappas) \
                     + s[0] - 2*s[1] > 0)
            if np.random.rand() < self.eps:
                return 1-ind
            else:
                return ind

    #%%
    pi_eps_greedy = ContPolicy(env=ContEnv, eps=epsilon, device=device)

    #%% Monte Carlo Value estimation of pi_eps_greedy
    MC_cfg={'episode_num':test_samp_size, 'verbose':False}
    pi_eps_greedy._to_cpu()
    ContEnv.params['offline'] = False
    reward_mean, Rewards, Initial_S, Initial_W = MC_evaluator(ContEnv, policy=pi_eps_greedy, config=MC_cfg, seed=0)
    Initial_W = _to_tensor(torch.tensor(Initial_W).view(-1,1), device=device) 
    Initial_S = _to_tensor(torch.tensor(Initial_S), device=device)

    #%% Training and evaluating performance
    ContEnv.params['offline'] = True
    train_cfg = {'episode_num':samp_size}
    pfqe_option={'gamma_f':'auto', 'n_gamma_hs':20, 'n_alphas':30, 'cv':5}
    for sim_num in range(100):
        with torch.no_grad():
            pi_eps_greedy._to_cpu()
            train_batch = batch_data_collector(ContEnv, train_cfg, policy = None, seed=sim_num)
            pi_eps_greedy._to_device()
            vpi_cont = fit_v0(
                Episodes          = train_batch, 
                action_space      = ContEnv.action_space, 
                observation_space = ContEnv.observation_space, 
                policy            = pi_eps_greedy, 
                option            = pfqe_option, 
                device            = device
                )
            pFQE_Rewards = vpi_cont(
                Initial_W,
                Initial_S
                )
            with open(file_path, 'a+') as csv_file:
                writer = csv.writer(csv_file, delimiter=',')
                writer.writerow([str(episode_len), str(samp_size), str(epsilon), str(abs(reward_mean - pFQE_Rewards.cpu().numpy().mean()))])
                csv_file.flush()
                os.fsync(csv_file.fileno())

if __name__=='__main__':
    main()
