import torch
import torch.nn as nn
import torch.nn.functional as F
from rotary_embedding_torch import RotaryEmbedding


class RMSNorm(nn.Module):
    def __init__(self, dimension, groups=1):
        super(RMSNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(dimension))
        self.groups = groups
        self.eps = 1e-5

    def forward(self, input):
        B, N, T = input.shape
        assert N % self.groups == 0

        input_float = input.reshape(B, self.groups, -1, T).float()
        input_norm = input_float * torch.rsqrt(input_float.pow(2).mean(-2, keepdim=True) + self.eps)

        return input_norm.type_as(input).reshape(B, N, T) * self.weight.reshape(1, -1, 1)


class RoformerAttention(nn.Module):
    def __init__(self, feature_dim, num_head=8, input_drop=0., attention_drop=0.):
        super(RoformerAttention, self).__init__()
        assert feature_dim % num_head == 0

        self.feature_dim = feature_dim
        self.num_head = num_head
        self.hidden_size = feature_dim // num_head
        self.attention_drop = attention_drop

        self.input_norm = RMSNorm(feature_dim)
        self.input_drop = nn.Dropout(input_drop)
        self.weight = nn.Conv1d(feature_dim, feature_dim * 3, 1, bias=False)
        self.output = nn.Conv1d(feature_dim, feature_dim, 1, bias=False)
        self.rotary_emb = RotaryEmbedding(dim=self.hidden_size)

    def forward(self, input):
        B, N, T = input.shape

        weight = self.weight(self.input_drop(self.input_norm(input)))
        weight = weight.reshape(B, self.num_head, self.hidden_size * 3, T).mT
        Q, K, V = torch.split(weight, self.hidden_size, dim=-1)

        Q = self.rotary_emb.rotate_queries_or_keys(Q)
        K = self.rotary_emb.rotate_queries_or_keys(K)

        dropout_p = self.attention_drop if self.training else 0.
        output = F.scaled_dot_product_attention(Q.contiguous(), K.contiguous(), V.contiguous(), dropout_p=dropout_p, is_causal=False)
        output = output.mT.reshape(B, N, T)
        output = self.output(output)

        return output


class FFN(nn.Module):
    def __init__(self, feature_dim):
        super(FFN, self).__init__()
        self.MLP = nn.Sequential(RMSNorm(feature_dim),
                                 nn.Conv1d(feature_dim, feature_dim * 8, 1, bias=False),
                                 nn.SiLU())
        self.MLP_output = nn.Conv1d(feature_dim * 4, feature_dim, 1, bias=False)

    def forward(self, input):
        gate, z = self.MLP(input).chunk(2, dim=1)
        output = input + self.MLP_output(F.silu(gate) * z)

        return output


class BiGRUFFN(nn.Module):
    def __init__(self, feature_dim):
        super(BiGRUFFN, self).__init__()
        self.norm = RMSNorm(feature_dim)
        self.gru = nn.GRU(feature_dim, feature_dim * 2, batch_first=True, bidirectional=True)
        self.output = nn.Linear(feature_dim * 4, feature_dim)

    def forward(self, input):
        output = self.norm(input).transpose(1, 2).contiguous()
        self.gru.flatten_parameters()
        output, _ = self.gru(output)
        output = self.output(output).transpose(1, 2).contiguous()
        output = input + output

        return output


class BandRoformerBlock(nn.Module):
    def __init__(self, feature_dim, num_head=8, input_drop=0., attention_drop=0.):
        super(BandRoformerBlock, self).__init__()
        self.attention = RoformerAttention(feature_dim, num_head, input_drop, attention_drop)
        self.ffn = FFN(feature_dim)

    def forward(self, input):
        B, nband, N, T = input.shape

        output = input.permute(0, 3, 2, 1).contiguous().reshape(B * T, N, nband)
        output = output + self.attention(output)
        output = self.ffn(output)
        output = output.reshape(B, T, N, nband).permute(0, 3, 2, 1).contiguous()

        return output


class TimeRoformerBlock(nn.Module):
    def __init__(self, feature_dim, num_head=8, input_drop=0., attention_drop=0.):
        super(TimeRoformerBlock, self).__init__()
        self.attention = RoformerAttention(feature_dim, num_head, input_drop, attention_drop)
        self.ffn = BiGRUFFN(feature_dim)

    def forward(self, input):
        B, nband, N, T = input.shape

        output = input.contiguous().reshape(B * nband, N, T)
        output = output + self.attention(output)
        output = self.ffn(output)
        output = output.reshape(B, nband, N, T)

        return output


class TSRoformerBlock(nn.Module):
    def __init__(self, feature_dim, num_head=8, input_drop=0., attention_drop=0.):
        super(TSRoformerBlock, self).__init__()
        self.band_block = BandRoformerBlock(feature_dim, num_head, input_drop, attention_drop)
        self.time_block = TimeRoformerBlock(feature_dim, num_head, input_drop, attention_drop)

    def forward(self, input):
        output = self.band_block(input)
        output = self.time_block(output)

        return output


class TSRoformer(nn.Module):
    def __init__(self, h):
        super(TSRoformer, self).__init__()
        self.feature_dim = h.feature_dim
        self.num_layers = h.num_layers
        self.num_head = h.num_heads
        self.input_drop = h.input_drop
        self.attention_drop = h.attention_drop

        self.blocks = nn.ModuleList([])
        for i in range(self.num_layers):
            self.blocks.append(TSRoformerBlock(self.feature_dim, self.num_head, self.input_drop, self.attention_drop))

    def forward(self, input):
        output = input
        for i in range(self.num_layers):
            output = self.blocks[i](output)

        return output
