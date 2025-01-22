import torch
import torch.nn as nn
from basemodel import MLP
from transformers_ import PositionalEncoding
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt

class AttentionDownsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AttentionDownsample, self).__init__()
        
    def forward(self, x):
        # 输入形状是 [batch, head, h, w]
        batch, head, h, w = x.size()
        
        x = nn.AvgPool2d(kernel_size=2, stride=2)(x)

        x = F.softmax(x, dim=-1)

        return x
    
# class AttentionDownsample(nn.Module):
#     def __init__(self, head):
#         super(AttentionDownsample, self).__init__()
#         self.conv_layers = nn.Sequential(
#             nn.Conv2d(head, head*2, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(head*2),
#             nn.LeakyReLU(),
#             nn.Conv2d(head*2, head*4, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(head*4),
#             nn.LeakyReLU(),
#             nn.Conv2d(head*4, head*4, kernel_size=3, stride=2, padding=1),
#             nn.BatchNorm2d(head*4),
#             nn.LeakyReLU(),
#             nn.Conv2d(head*4, head*2, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(head*2),
#             nn.LeakyReLU(),
#             nn.Conv2d(head*2, head, kernel_size=3, stride=1, padding=1),
#         )
        
#     def forward(self, x):
#         assert x.dim() == 4, "Input must be 4-dimensional"
#         # x 的形状为 (batch_size, head, 8, 8)
#         out = self.conv_layers(x)
#         # 归一化到 [0, 1]
#         out = F.softmax(out, dim=-1)
#         return out

class CDA(nn.Module):
    def __init__(self, d_model, key_len, num_heads=4):
        super(CDA, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.learnable_m = nn.Parameter(torch.randn(8, 1, 20, d_model), requires_grad=True)
        self.learnable_s = nn.Parameter(torch.randn(8, 1, 20, d_model), requires_grad=True)

        d_model = d_model * 2
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.pe = PositionalEncoding(d_model, batch_first=False)
        # Linear layers for query, key, and value projections
        self.query = nn.Linear(d_model, d_model)

        self.key1 = nn.Linear(d_model, d_model)
        self.value1 = nn.Linear(d_model // 2, d_model)

        self.key2 = nn.Linear(d_model, d_model)
        self.value2 = nn.Linear(d_model // 2, d_model)
        # 用卷积 将160*160的注意力矩阵转换为80*80的注意力矩阵
        self.attention_downsample1 = AttentionDownsample(4, 4)
        self.attention_downsample2 = AttentionDownsample(4, 4)

        # Final projection
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model * 2, d_model*4),
            nn.ReLU(inplace=True),
            nn.Linear(d_model * 4, d_model),
        )

    
    def add_positional_encoding(self, x):
        T, N, K, D = x.size()
        x = x.reshape(T, N*K, D)
        x = self.pe(x)
        x = x.reshape(T, N, K, D)
        return x

    def forward(self, m0, s0, m1, s1, k1, m2, s2, k2):
        T, N, K, D = m0.size()
        T_sub = k1.size(0)

        ms0 = torch.cat([m0, s0], dim=-1)
        ms0 = self.add_positional_encoding(ms0)
        ms1 = torch.cat([m1, s1], dim=-1)
        ms1 = self.add_positional_encoding(ms1)
        ms2 = torch.cat([m2, s2], dim=-1)
        ms2 = self.add_positional_encoding(ms2)

        # fake ms
        ms1_fake = torch.cat([self.learnable_m.expand_as(m1), s1], dim=-1)
        ms1_fake = self.add_positional_encoding(ms1_fake)
        ms2_fake = torch.cat([m2, self.learnable_s.expand_as(s2)], dim=-1)
        ms2_fake = self.add_positional_encoding(ms2_fake) # [8,N,K,2D]

        # cal query
        ms0 = ms0.transpose(1, 2).reshape(T*K, N, D*2) # [T*K, N, 2D]
        query = self.query(ms0).view(T*K, N, self.num_heads, self.head_dim).permute(1, 2, 0, 3) # [N, 4, T*K, head_dim]
        # cal key
        key1 = self.key1(ms1).view(T*K, N, self.num_heads, self.head_dim).permute(1, 2, 0, 3) # [N, 4, T*K, head_dim]
        key1_fake = self.key1(ms1_fake).view(T*K, N, self.num_heads, self.head_dim).permute(1, 2, 0, 3) # [N, 4, T*K, head_dim]
        key2 = self.key2(ms2).view(T*K, N, self.num_heads, self.head_dim).permute(1, 2, 0, 3) # [N, 4, T*K, head_dim]
        key2_fake = self.key2(ms2_fake).view(T*K, N, self.num_heads, self.head_dim).permute(1, 2, 0, 3) # [N, 4, T*K, head_dim]
        # cal attention
        atten1 = torch.matmul(query, key1.transpose(-2, -1)) / (self.head_dim ** 0.5) # [N, 4, T*K, T*K]
        atten1_fake = torch.matmul(query, key1_fake.transpose(-2, -1)) / (self.head_dim ** 0.5) # [N, 4, T*K, T*K]
        atten2 = torch.matmul(query, key2.transpose(-2, -1)) / (self.head_dim ** 0.5) # [N, 4, T*K, T*K]
        atten2_fake = torch.matmul(query, key2_fake.transpose(-2, -1)) / (self.head_dim ** 0.5) # [N, 4, T*K, T*K]

        # plt.figure(0)
        # for i in range(4):
        #     plt.subplot(2, 2, i+1)
        #     plt.imshow(atten1[0, i].detach().cpu().numpy())
        
        # plt.figure(1)
        # for i in range(4):
        #     plt.subplot(2, 2, i+1)
        #     plt.imshow(atten1_fake[0, i].detach().cpu().numpy())

        # subtract attention
        kept_atten1 = torch.softmax(torch.softmax(atten1, dim=-1) - torch.softmax(atten1_fake, dim=-1), dim=-1) # [N, 4, T*K, T*K]
        kept_atten2 = torch.softmax(torch.softmax(atten2, dim=-1) - torch.softmax(atten2_fake, dim=-1), dim=-1)
        
        # downsample attention
        kept_atten1_down = self.attention_downsample1(kept_atten1)
        kept_atten2_down = self.attention_downsample2(kept_atten2)

        # plt.figure(2)
        # for i in range(4):
        #     plt.subplot(2, 2, i+1)
        #     plt.imshow(kept_atten1[0, i].detach().cpu().numpy())

        # plt.figure(3)
        # for i in range(4):
        #     plt.subplot(2, 2, i+1)
        #     plt.imshow(kept_atten1_down[0, i].detach().cpu().numpy())
        # plt.show()
        # plt.close()


        # cal value
        value1 = self.value1(k1).transpose(1, 2).reshape(T_sub*K, N, self.num_heads, self.head_dim).permute(1, 2, 0, 3) # [T_sub*K, N, 4, head_dim]
        value2 = self.value2(k2).transpose(1, 2).reshape(T_sub*K, N, self.num_heads, self.head_dim).permute(1, 2, 0, 3)
        # cal output
        output1 = torch.matmul(kept_atten1_down, value1).permute(2, 0, 1, 3)
        output2 = torch.matmul(kept_atten2_down, value2).permute(2, 0, 1, 3)

        # concatenate heads and apply final output projection
        output1 = output1.reshape(T_sub, K, N, self.d_model)
        output2 = output2.reshape(T_sub, K, N, self.d_model)
        output = torch.cat([output1, output2], dim=-1)
        output = self.feed_forward(output)
        return output

    
# class TF(nn.Module):
#     def __init__(self, embed_size, heads, need_pe=False, batch_first=True, dropout=0, forward_expansion=4):
#         super(TF, self).__init__()
        
#         self.heads = heads
#         self.need_pe = need_pe
#         self.attention = nn.MultiheadAttention(embed_size, heads, dropout=dropout, batch_first=batch_first)

#         self.norm1 = nn.LayerNorm(embed_size)
#         self.norm2 = nn.LayerNorm(embed_size)
#         self.positional_encoding = PositionalEncoding(embed_size, batch_first=batch_first)

#         self.feed_forward = nn.Sequential(
#             nn.Linear(embed_size, forward_expansion * embed_size),
#             nn.ReLU(inplace=True),
#             nn.Linear(forward_expansion * embed_size, embed_size),
#         )
        
#         self.dropout = nn.Dropout(dropout)

#     def forward(self, query, kv, mask=None):
#         if self.need_pe:
#             query, kv = self.positional_encoding(query), self.positional_encoding(kv)

#         if mask is not None:
#             mask = mask.repeat(self.heads, 1, 1)

#         attention_output = self.attention(query, kv, kv, attn_mask=mask)[0]

#         x = self.dropout(self.norm1(attention_output + query))
        
#         forward_output = self.feed_forward(x)
        
#         output = self.dropout(self.norm2(forward_output + x))
#         return output

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    model = CDA(64, 4).cuda()
    m0 = torch.randn(8, 700, 20, 64).cuda()
    k1 = torch.randn(4, 700, 20, 64).cuda()
    out = model(m0, m0, m0, m0, k1, m0, m0, k1)
    print(out.shape)