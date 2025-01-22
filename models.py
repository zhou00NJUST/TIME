from numpy import dot
from basemodel import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from transformers_ import TransformerBlock
from my_decoder import Conv_Decoder
from torch_geometric.utils import to_dense_adj, add_self_loops
from causal_disentanglement import CDA

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def gen_tot_mask(batch_split, edge_pair, N_batch, device):
    mask_tot = torch.ones(N_batch, N_batch, device=device).bool()
    for i, (left, right) in enumerate(batch_split):
        left = left.item()
        right = right.item()
        now_pair = edge_pair[i]
        N = right - left
        if len(now_pair) != 0:
            edge_pair_now = now_pair.transpose(0, 1)
        else:
            edge_pair_now = now_pair
        edge_index_with_self_loops, edge_weight = add_self_loops(edge_pair_now, num_nodes=N)

        attn_mask = (to_dense_adj(edge_index_with_self_loops, max_num_nodes=N).squeeze(0)  ).bool() # [N, N]
        mask_tot[left:right, left:right] = ~attn_mask # 存在边的变为0(不掩盖)
    return mask_tot

class LaplaceNLLLoss(nn.Module):

    def __init__(self,
                 eps: float = 1e-6,
                 reduction: str = 'mean') -> None:
        super(LaplaceNLLLoss, self).__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(self,
                pred: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        loc, scale = pred.chunk(2, dim=-1)
        scale = scale.clone()
        # print("scale",scale.shape,"loc",loc.shape)
        with torch.no_grad():
            scale.clamp_(min=self.eps)
        nll = torch.log(2 * scale) + torch.abs(target - loc) / scale
        # print("nll", nll.shape)
        if self.reduction == 'mean':
            return nll.mean()
        elif self.reduction == 'sum':
            return nll.sum()
        elif self.reduction == 'none':
            return nll
        else:
            raise ValueError('{} is not a valid value for reduction'.format(self.reduction))
        
class TemporalEncoder(nn.Module):
    def __init__(self, args):
        super(TemporalEncoder, self).__init__()
        self.args = args
        self.t_encoder = Temperal_Encoder(args)
        self.conv_to12 = Conv_Decoder(args)
    def forward(self, x):
        out = self.t_encoder(x)
        out12 = self.conv_to12(out)
        return out, out12

class SpatialEncoder(nn.Module):
    def __init__(self, args):
        super(SpatialEncoder, self).__init__()
        self.args = args
        self.s_encoder = TransformerBlock(args.hidden_size, 4)
    def forward(self, x, mask):
        out = self.s_encoder(x, x, mask.repeat(x.size(0), 1, 1))
        return out # [8,N,D]

class KeyPointEncoder(nn.Module):
    def __init__(self, args):
        super(KeyPointEncoder, self).__init__()
        self.args = args
        self.mlp = MLP(2, self.args.hidden_size)
        self.gru = nn.GRU(self.args.hidden_size, self.args.hidden_size, batch_first=True)
    def forward(self, x):
        x_enc = self.mlp(x)
        out, hn = self.gru(x_enc)
        return out

class TransformerDecoder(nn.Module):
    def __init__(self, args):
        super(TransformerDecoder, self).__init__()
        self.args = args
        self.num_modes = 20
        self.hidden_size = args.hidden_size
        self.key_steps = 4
        
        self.multi_proj = MultimodalProj(args)

        self.gru = nn.GRU(self.hidden_size, self.hidden_size, batch_first=True)
        
        self.spatial_fusion = TransformerBlock(self.hidden_size, heads=4, need_pe=True, batch_first=True)
        self.memory_fusion = TransformerBlock(self.hidden_size, heads=4, need_pe=True, batch_first=True)
        self.modal_attn = TransformerBlock(self.hidden_size, heads=4, need_pe=True, batch_first=False)

        self.loc = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_size, 2))
        self.scale = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_size, 2))
    def forward(self, x_encoding, x_encoding_12, x_social, key_points_encoding, multi_memo = False):
        x_encoding_12 = self.multi_proj(x_encoding_12) # [20N,12,D]

        social_token_rep = x_social.repeat(self.num_modes,1,1) # [20N,1,D]
        
        if not multi_memo:
            # 训练时, 用的真实的key_points, 需要repeat成20个
            key_points_token_rep = key_points_encoding.repeat(self.num_modes,1,1) # [20N,4,D]
        else:
            # 测试时, 用的是检索的20条记忆的key_points
            key_points_token_rep = key_points_encoding # [20N,4,D]
        
        # memory modal attention
        key_points_token_rep = key_points_token_rep.reshape(self.num_modes, -1, self.hidden_size) # [20, N*4, D]
        key_points_token_rep_modal = key_points_token_rep # self.modal_attn(key_points_token_rep, key_points_token_rep) # [20, N*4, D]
        key_points_token_rep_modal = key_points_token_rep_modal.reshape(-1, self.key_steps, self.hidden_size) # [20N, 4, D]

        out_st = self.spatial_fusion(x_encoding_12, social_token_rep)
        out_stm = self.memory_fusion(out_st, key_points_token_rep_modal)

        out = self.gru(out_stm)[0]
        loc = self.loc(out).view(self.num_modes, -1, 12, 2)
        scale = (F.elu_(self.scale(out), alpha=1.0) + 1.0 + 1e-3).view(self.num_modes, -1, 12, 2)
        return (loc, scale), out


# class MyDiffusion(nn.Module):
#     def __init__(self, args, max_T):
#         super().__init__()
#         self.hidden_size = args.hidden_size
#         self.diffnet = TransformerConcatLinear(point_dim=2, context_dim=self.hidden_size)
#         self.diffusion = DiffusionTraj(
#             net = self.diffnet,
#             var_sched = VarianceSchedule(
#                 num_steps=max_T,
#                 beta_1=1e-4,
#                 beta_T=2e-2,
#                 mode='linear'
#             )
#         )
        
#         self.t_encoder = TemporalEncoder(args)
#         self.s_encoder = SpatialEncoder(args)
#         self.gru = nn.GRU(self.hidden_size, self.hidden_size, batch_first=True)
    
#         self.multi_proj = MultimodalProj(args)
    
#     def gen_codition(self, train_x, mask_tot):
#         x_encoding, x_encoding_12 = self.t_encoder(train_x)
#         x_social = self.s_encoder(x_encoding.transpose(0, 1), mask_tot)
#         obs_condition = self.gru(x_encoding_12, (x_encoding[:, -1] + x_social))[0] # [B,12,D]
        
#         return obs_condition
    
#     def generate(self, train_x, mask_tot, memory_samples):
#         condition = self.gen_codition(train_x, mask_tot)
#         condition = self.multi_proj(condition).reshape(20,-1,12,self.hidden_size) # [20, B, 12, D]
        
#         predicted_y_pos = self.diffusion.sample(condition, memory_samples)
#         return predicted_y_pos
    
#     def get_loss(self, train_y_gt, train_x, mask_tot, memory_samples):
#         condition = self.gen_codition(train_x, mask_tot) # [N, 12, D]
#         condition = self.multi_proj(condition).reshape(20,-1,12,self.hidden_size) # [20, B, 12, D]
#         mse_loss = self.diffusion.get_loss(train_y_gt, condition, memory_samples)
#         return mse_loss

class MultiTrajEncoder(nn.Module):
    def __init__(self, hidden_size):
        super(MultiTrajEncoder, self).__init__()

        self.hidden_size = hidden_size
        self.gru = nn.GRU(self.hidden_size, self.hidden_size, batch_first=False)
        self.mlp = MLP(2, self.hidden_size)
    
    def forward(self, x, need_full = False):
        T, N, K, D = x.shape
        x = x.reshape(T, -1, D) # [T,NK,2]
        x = self.mlp(x)
        full_encoding, out = self.gru(x)
        out = out.reshape(N, K, self.hidden_size)
        if need_full:
            full_encoding = full_encoding.reshape(T, N, K, self.hidden_size)
            return full_encoding, out
        else:
            return out

class MDTraj(nn.Module):
    def __init__(self, args):
        super(MDTraj, self).__init__()
        self.args = args

        self.num_modes = 20
        self.hidden_size = args.hidden_size
        self.key_points = args.key_points
        self.key_len = len(self.key_points)

        self.t_encoder = TemporalEncoder(args)
        self.s_encoder = SpatialEncoder(args)
        self.key_point_encoder = KeyPointEncoder(args)
        self.tf_decoder = TransformerDecoder(args)

        self.traj_encoder = MultiTrajEncoder(self.hidden_size)
        self.key_points_encoder = MultiTrajEncoder(self.hidden_size)
        self.curr_obs_encoder = MultiTrajEncoder(self.hidden_size)
        

        self.cda = CDA(self.hidden_size, key_len=self.key_len)
        self.key_out = MLP(self.hidden_size * 2, self.hidden_size)
        
        # self.my_diff = MyDiffusion(args, max_T=300)
    
    def forward(self, inputs, get_encodings_only=False):
        batch_abs_gt, batch_norm_gt, key_points, shitf_values, max_values, batch_split, edge_pair = inputs
        device = torch.device(batch_abs_gt.device)

        # extract data
        # batch_class = batch_abs_gt[0,:,-1].long()
        batch_abs_gt = batch_abs_gt[:,:,:2]
        self.batch_norm_gt = batch_norm_gt
        
        offset = batch_norm_gt[1:self.args.obs_length, :, :] - batch_norm_gt[:self.args.obs_length-1, :, :] #[H, N, 2]
        position = batch_norm_gt[:self.args.obs_length, :, :] # [H, N, 2]
        pad_offset = torch.zeros_like(position, device=device)
        pad_offset[1:, :, :] = offset
        train_x_ = torch.cat((position, pad_offset), dim=2) # [H, N, 4]
        
        train_x = train_x_.permute(1, 2, 0) # [N, 4, H]
        train_y_gt = batch_norm_gt[self.args.obs_length:, :, :].transpose(0, 1)
        self.pre_obs = batch_norm_gt[1:self.args.obs_length]
        N_batch = train_x.size(0) # agent number N
        # gen past social mask
        mask_tot = gen_tot_mask(batch_split, edge_pair, N_batch, device)

        key_points = key_points.transpose(0, 1) # [N,4,2]
        
        # temporal modeling
        x_encoding, x_encoding_12 = self.t_encoder(train_x) # [N, H=8, D], [N, 12, D]
        x_social = self.s_encoder(x_encoding.transpose(0, 1), mask_tot).transpose(0, 1) # [N, 8, D]
           

        # key_point_encoding = self.key_point_encoder(train_y_gt[:, self.key_points]) # [N, D]
        key_point_encoding = self.key_point_encoder(key_points) # [N, D]
        key_point_encoding = F.dropout(key_point_encoding, p=0.1, training=self.training) # dropout while training to avoid too much dependence on key points
        pred_trajs, _ = self.tf_decoder(x_encoding, x_encoding_12, x_social, key_point_encoding)

        x_encoding_squeezed = x_encoding[:, -1]
        if get_encodings_only:
            return x_encoding_squeezed, x_social.squeeze(), key_point_encoding.squeeze()
        
        if self.training:
            if self.args.train_phase == 'gen_memory':
                loss = self.mdn_loss(pred_trajs, train_y_gt)
                return loss
            elif self.args.train_phase == 'train_adaptor':
                adapted_key = self.search_memory(position, x_social.squeeze())
                adapted_key = adapted_key.reshape(-1,self.key_len,self.hidden_size) # [20N,4,D]
                
                pred_trajs_memo, _ = self.tf_decoder(x_encoding, x_encoding_12, x_social, adapted_key, multi_memo=True)
                
                loss = self.mdn_loss(pred_trajs_memo, train_y_gt)
                
                return loss
                
            elif self.args.train_phase == 'train_diffusion':
                pass
                # adapted_key = self.search_memory(x_encoding_squeezed.squeeze(), x_social.squeeze()) # [B,K,D]
                # adapted_key = adapted_key.reshape(-1,1,self.hidden_size) # [20N,1,D]
                # pred_trajs_memo, _ = self.tf_decoder(x_encoding, x_encoding_12, x_social, adapted_key, multi_memo=True)

                # diff_loss = self.my_diff.get_loss(train_y_gt, train_x, mask_tot, pred_trajs_memo.detach())
                # return diff_loss
        else:
            if self.args.train_phase == 'gen_memory':
                final_samples = pred_trajs[0]
                return self.keep_best(final_samples, train_y_gt)

            adapted_key = self.search_memory(position, x_social.squeeze())
            adapted_key = adapted_key.reshape(-1,self.key_len,self.hidden_size) # [20N,4,D]
            
            pred_trajs_memo, _ = self.tf_decoder(x_encoding, x_encoding_12, x_social, adapted_key, multi_memo=True)
            
            if self.args.train_phase == 'train_adaptor':
                final_samples = pred_trajs_memo[0]
            # elif self.args.train_phase == 'train_diffusion':
            #     final_samples = self.my_diff.generate(train_x, mask_tot, pred_trajs_memo)
            return self.keep_best(final_samples, train_y_gt)

    def search_memory(self, obs_traj, x_social):
        t_sim = torch.norm(obs_traj.unsqueeze(2) - self.memory_t.unsqueeze(1), p=2, dim=-1).mean(0) # [N, memory size]
        idx_based_on_t = torch.sort(t_sim, dim=-1, descending=False)[1][:, :self.num_modes] # [N, K], searched by temporal
        s_sim = cosine_similarity(x_social, self.memory_s)
        idx_based_on_s = torch.sort(s_sim, dim=-1, descending=True)[1][:, :self.num_modes] # [N, K], searched by social

        closest_traj_t = self.traj_encoder(self.memory_t[:, idx_based_on_t], need_full=True)[0] 
        closest_traj_s = self.traj_encoder(self.memory_t[:, idx_based_on_s], need_full=True)[0]

        closest_key_points_t = self.key_points_encoder(self.memory_key[:, idx_based_on_t], need_full=True)[0] # [4, N, K, D]
        closest_key_points_s = self.key_points_encoder(self.memory_key[:, idx_based_on_s], need_full=True)[0]

        obs_encoding = self.curr_obs_encoder(obs_traj.unsqueeze(2), need_full=True)[0].repeat(1,1,self.num_modes,1) # [8,N,K,D]
        x_social_ = x_social.transpose(0, 1).unsqueeze(2).repeat(1,1,self.num_modes,1) # [8,N,K,D]
        # casual disentanglement adaptor
        m0 = obs_encoding # [T, N, K, D]
        s0 = x_social_
        m1 = closest_traj_t
        s1 = self.memory_s[:, idx_based_on_t]
        k1 = closest_key_points_t # [T_sub, N, K, D]
        m2 = closest_traj_s
        s2 = self.memory_s[:, idx_based_on_s]
        k2 = closest_key_points_s
        
        cda_out = self.cda(m0, s0, m1, s1, k1, m2, s2, k2) # [T_sub, N, K, 2D]
        

        cda_out = self.key_out(cda_out)

        return cda_out

    
    def mdn_loss(self, y_prime, y):
        batch_size = y.shape[0]
        
        out_mu, out_sigma = y_prime 
        y_hat = torch.cat((out_mu, out_sigma), dim=-1)
        
        l2_norm = (torch.norm(out_mu - y, p=2, dim=-1) ).sum(dim=-1)   # [F, N]
        best_mode = l2_norm.argmin(dim=0)
        y_hat_best = y_hat[best_mode, torch.arange(batch_size)]
        reg_loss = LaplaceNLLLoss()(y_hat_best, y)
        
        return reg_loss
    
    def keep_best(self, pred, y):
        batch_size=y.shape[0]
        
        loc = pred

        full_pre_tra = []
        # best ADE
        l2_norm = (torch.norm(loc - y, p=2, dim=-1) ).sum(dim=-1)   # [F, N]
        best_mode = l2_norm.argmin(dim=0) # N个中每一个20中的最小的坐标
        sample_k = loc[best_mode, torch.arange(batch_size)].permute(1, 0, 2)  #[H, N, 2]
        full_pre_tra.append(torch.cat((self.pre_obs,sample_k), axis=0))
        # best FDE
        l2_norm_FDE = (torch.norm(loc[:,:,-1,:] - y[:,-1,:], p=2, dim=-1) )  # [F, N]
        best_mode = l2_norm_FDE.argmin(dim=0)
        sample_k = loc[best_mode, torch.arange(batch_size)].permute(1, 0, 2)  #[H, N, 2]
        full_pre_tra.append(torch.cat((self.pre_obs,sample_k), axis=0))
        return full_pre_tra

class MultimodalProj(nn.Module):
    def __init__(self, args):
        super(MultimodalProj, self).__init__()
        self.args = args
        self.num_modes = 20
        self.hidden_size = args.hidden_size
        self.multimodal_proj = nn.Sequential(
            nn.Linear(self.hidden_size , self.num_modes * self.hidden_size),
            nn.LayerNorm(self.num_modes * self.hidden_size),
            nn.ReLU(inplace=True))
    def forward(self, x):
        x_rep = self.multimodal_proj(x)
        x_rep = x_rep.view(-1, x.size(1), self.num_modes, self.hidden_size)
        x_rep = x_rep.permute(2, 0, 1, 3)
        x_rep = x_rep.reshape(self.num_modes*x.size(0), -1, self.hidden_size)
        return x_rep

def cosine_similarity(x, y):
    # 计算点积
    # x: N,8,D; y: 8,M,D --> attention: 8, N, M
    dot_product = torch.einsum('ntd,tmd->tnm', x, y)
    
    # dot_product = torch.mm(x, y.transpose(-1, -2))
    # 计算范数
    norm_x = x.norm(dim=-1).view(8, -1, 1)  # N x 1
    norm_y = y.norm(dim=-1).view(8, -1, 1)  # M x 1
    # 计算相似度矩阵
    similarity_matrix = dot_product / (norm_x * norm_y.transpose(-1, -2))
    return similarity_matrix.mean(0) # average over 8
