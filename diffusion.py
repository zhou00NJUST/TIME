import torch
import torch.nn.functional as F
from torch.nn import Module
import numpy as np
import torch.nn as nn
from common import ConcatSquashLinear
import math
from copy import deepcopy
from transformers_ import TransformerBlock

std_scale = 1 / 20.0
class VarianceSchedule(Module):
    def __init__(self, num_steps, mode='linear',beta_1=1e-4, beta_T=5e-2,cosine_s=8e-3):
        super().__init__()
        assert mode in ('linear', 'cosine')
        self.num_steps = num_steps
        self.beta_1 = beta_1
        self.beta_T = beta_T
        self.mode = mode

        if mode == 'linear':
            betas = torch.linspace(beta_1, beta_T, steps=num_steps)
        elif mode == 'cosine':
            timesteps = (
            torch.arange(num_steps + 1) / num_steps + cosine_s
            )
            alphas = timesteps / (1 + cosine_s) * math.pi / 2
            alphas = torch.cos(alphas).pow(2)
            alphas = alphas / alphas[0]
            betas = 1 - alphas[1:] / alphas[:-1]
            betas = betas.clamp(max=0.999)

        betas = torch.cat([torch.zeros([1]), betas], dim=0)     # Padding

        alphas = 1 - betas
        log_alphas = torch.log(alphas)
        for i in range(1, log_alphas.size(0)):  # 1 to T
            log_alphas[i] += log_alphas[i - 1]
        alpha_bars = log_alphas.exp()

        sigmas_flex = torch.sqrt(betas)
        sigmas_inflex = torch.zeros_like(sigmas_flex)
        for i in range(1, sigmas_flex.size(0)):
            sigmas_inflex[i] = ((1 - alpha_bars[i-1]) / (1 - alpha_bars[i])) * betas[i]
        sigmas_inflex = torch.sqrt(sigmas_inflex)

        self.betas = betas
        self.alphas = alphas
        self.alpha_bars = alpha_bars
        self.sigmas_flex = sigmas_flex
        self.sigmas_inflex = sigmas_inflex

    def uniform_sample_t(self, batch_size):
        ts = np.random.choice(np.arange(1, self.num_steps+1), batch_size)
        return ts.tolist()

    def get_sigmas(self, t, flexibility):
        assert 0 <= flexibility and flexibility <= 1
        sigmas = self.sigmas_flex[t] * flexibility + self.sigmas_inflex[t] * (1 - flexibility)
        return sigmas

class DiffusionTraj(Module):

    def __init__(self, net, var_sched:VarianceSchedule):
        super().__init__()
        self.net = net
        self.var_sched = var_sched

    def get_loss(self, x_0, condition, memory_samples, t=None):
        num_modes = memory_samples.size(0)
        hidden_size = condition.size(-1)

        condition = condition.reshape(-1, 12, hidden_size)
        memory_samples = memory_samples.reshape(-1, 12, 2)
        x_0 = x_0.repeat(num_modes, 1, 1)

        batch_size, _, point_dim = x_0.size()
        if t == None:
            t = self.var_sched.uniform_sample_t(batch_size)

        # get beta for T
        beta = self.var_sched.betas[t].cuda()
        beta = torch.tensor(t).cuda() / self.var_sched.num_steps

        # get e_T
        alpha_bar = self.var_sched.alpha_bars[t]
        c0 = torch.sqrt(alpha_bar).view(-1, 1, 1).cuda()
        c1 = torch.sqrt(1 - alpha_bar).view(-1, 1, 1).cuda()
        e_rand = torch.randn_like(x_0).cuda() * std_scale

        x_T = c0 * x_0 + c1 * e_rand + (1 - c0) * memory_samples
        e_theta = self.net(x_T, beta=beta, condition=condition, pred_hatx0=memory_samples)
        loss = F.mse_loss(e_theta, e_rand)
        return loss

    def sample(self, condition, memory_samples, point_dim=2, ret_traj=False):
        dev = condition.device
        traj_list = []

        
        hidden_size = condition.size(-1)
        pred_len = condition.size(2)
        # x_T = torch.zeros_like(memory_samples) * std_scale + memory_samples # (20, B, 12, 2)
        x_T = memory_samples
        
        num_ped = x_T.size(1)
        x_T = x_T.view(-1, pred_len, point_dim) # [20B,12,2]

        condition = condition.reshape(-1, 12, hidden_size)
        memory_samples = memory_samples.reshape(-1, 12, 2)
        
        batch_size = x_T.size(0)

        traj = {self.var_sched.num_steps: x_T}

        flexibility = 0.0
        sampling = 'ddim'
        # ddpm and ddim sampler
        stride = 10
        for t in range(self.var_sched.num_steps, 0, -stride):
            x_t = traj[t]
            
            if sampling == "ddpm":
                z = torch.randn_like(x_T) * std_scale if t > 1 else torch.zeros_like(x_T)
                alpha = self.var_sched.alphas[t]
                alpha_bar = self.var_sched.alpha_bars[t]
                beta = self.var_sched.betas[[t]*batch_size].cuda()

                e_theta = self.net(x_t, beta=beta, condition=condition, pred_hatx0=memory_samples)
                c0 = 1.0 / torch.sqrt(alpha)

                c1 = (1 - alpha) / torch.sqrt(1 - alpha_bar)
                sigma = self.var_sched.get_sigmas(t, flexibility)
                x_next = c0 * (x_t - c1 * e_theta - (1-alpha.sqrt()) * memory_samples) + sigma * z

            elif sampling == "ddim":
                alpha_bar = self.var_sched.alpha_bars[t]

                beta = self.var_sched.betas[[t]*batch_size].cuda()
                beta = torch.ones_like(beta) * t / self.var_sched.num_steps

                e_theta = self.net(x_t, beta=beta, condition=condition, pred_hatx0=memory_samples)
                alpha_bar_next = self.var_sched.alpha_bars[t-stride].cuda()
                x0_t = (x_t - e_theta * (1 - alpha_bar).sqrt() - (1-alpha_bar.sqrt()) * memory_samples) / alpha_bar.sqrt()
                x_next = alpha_bar_next.sqrt() * x0_t + (1 - alpha_bar_next).sqrt() * e_theta + (1-alpha_bar_next.sqrt()) * memory_samples 
            else:
                exit()
            
            traj[t-stride] = x_next.detach()     # Stop gradient and save trajectory.
            if t-stride == 0:
                break
            # traj[t] = traj[t].cpu()         # Move previous output to CPU memory.
            if not ret_traj:
                del traj[t]
                
        if ret_traj:
            pass
        else:
            traj_list.append(traj[0])

        # return torch.stack(traj_list)
        return traj_list[0].view(-1, num_ped, 12, 2)



class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()

        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)

class MLP_(nn.Module):
    def __init__(self, hidden_size, out_size):
        super(MLP_, self).__init__()
        self.hidden_size = hidden_size
        self.fc = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_size, out_size))
    def forward(self, x):
        return self.fc(x)


class TransformerConcatLinear(Module):
    def __init__(self, point_dim, context_dim):
        super().__init__()
        self.hidden_size = context_dim
        self.pred_len = 12
        self.concat1 = ConcatSquashLinear(2,context_dim, 2*context_dim+3)
        self.transformer_encoder1 = TransformerBlock(context_dim, heads=4, need_pe=True)

        self.concat2 = ConcatSquashLinear(context_dim, 2*context_dim, 2*context_dim+3)
        self.transformer_encoder2 = TransformerBlock(2*context_dim, heads=4, need_pe=True)

        self.concat3 = ConcatSquashLinear(2*context_dim, 4*context_dim, 2*context_dim+3)
        self.transformer_encoder3 = TransformerBlock(4*context_dim, heads=4, need_pe=True)

        self.concat4 = ConcatSquashLinear(4*context_dim, 4*context_dim, 2*context_dim+3)
        self.transformer_encoder4 = TransformerBlock(4*context_dim, heads=4, need_pe=True)

        self.concat5 = ConcatSquashLinear(4*context_dim, 2*context_dim, 2*context_dim+3)
        self.transformer_encoder5 = TransformerBlock(2*context_dim, heads=4, need_pe=True)

        self.concat6 = ConcatSquashLinear(2*context_dim, context_dim, 2*context_dim+3)
        self.transformer_encoder6 = TransformerBlock(context_dim, heads=4, need_pe=True)

        self.linear = ConcatSquashLinear(context_dim, 2, 2*context_dim+3)
        self.x0_up = MLP_(point_dim, context_dim)


    def forward(self, x, beta, condition, pred_hatx0):
        batch_size = x.size(0)
        
        beta = beta.view(batch_size, 1, 1)          # (B, 1, 1)
        condition = condition.view(batch_size, -1, self.hidden_size)   # (B, 12, F)

        time_emb = torch.cat([beta, torch.sin(beta), torch.cos(beta)], dim=-1)  # (B, 1, 3)
        time_emb = time_emb.repeat(1,self.pred_len,1) # [B, 12, 3]
        ctx_emb = torch.cat([time_emb, condition], dim=-1)    # (B, 1, F+3)

        pred_x0_encoding = self.x0_up(pred_hatx0)
        ctx_emb = torch.cat( [ctx_emb, pred_x0_encoding], dim=-1) # [B,12,2F+3]

        x = self.concat1(ctx_emb, x)
        trans1 = self.transformer_encoder1(x, x)

        trans2 = self.concat2(ctx_emb, trans1)
        trans2 = self.transformer_encoder2(trans2, trans2)

        trans3 = self.concat3(ctx_emb, trans2)
        trans3 = self.transformer_encoder3(trans3, trans3)

        trans4 = self.concat4(ctx_emb, trans3)
        trans4 = self.transformer_encoder4(trans4, trans4) + trans3

        trans5 = self.concat5(ctx_emb, trans4)
        trans5 = self.transformer_encoder5(trans5, trans5) + trans2

        trans6 = self.concat6(ctx_emb, trans5)
        trans6 = self.transformer_encoder6(trans6, trans6) + trans1

        return self.linear(ctx_emb, trans6)

