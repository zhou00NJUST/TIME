import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len=1000, batch_first=True):
        super(PositionalEncoding, self).__init__()
        positional_encodings = torch.zeros(max_seq_len, d_model)
        positions = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        
        positional_encodings[:, 0::2] = torch.sin(positions * div_term)
        positional_encodings[:, 1::2] = torch.cos(positions * div_term)
                
        self.register_buffer('positional_encodings', positional_encodings.unsqueeze(0))
        self.batch_first = batch_first

    def forward(self, x):
        if not self.batch_first:
            x = x.transpose(0, 1)
        
        x = x + self.positional_encodings[:, :x.size(1), :]
        
        if not self.batch_first:
            x = x.transpose(0, 1)

        return x


class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, need_pe=False, batch_first=True, dropout=0, forward_expansion=4):
        super(TransformerBlock, self).__init__()
        
        self.heads = heads
        self.need_pe = need_pe
        self.attention = nn.MultiheadAttention(embed_size, heads, dropout=dropout, batch_first=batch_first)

        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.positional_encoding = PositionalEncoding(embed_size, batch_first=batch_first)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(inplace=True),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, kv, mask=None):
        if self.need_pe:
            query, kv = self.positional_encoding(query), self.positional_encoding(kv)

        if mask is not None:
            mask = mask.repeat(self.heads, 1, 1)

        attention_output = self.attention(query, kv, kv, attn_mask=mask)[0]

        x = self.dropout(self.norm1(attention_output + query))
        
        forward_output = self.feed_forward(x)
        
        output = self.dropout(self.norm2(forward_output + x))
        return output

if __name__ == '__main__':
    model = TransformerBlock(512, 8, need_pe=True, batch_first=True)
    x = torch.rand(10, 32, 512).transpose(0, 1)
    y = torch.rand(10, 32, 512).transpose(0, 1)
    output = model(x, y)
    print(output.shape)