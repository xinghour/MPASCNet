import torch
import torch.nn as nn
import torch.nn.functional as F
from models.band_split.band_bias import BandBias


class RMSNorm(nn.Module):
    def __init__(self, dimension, groups=1):
        super(RMSNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(dimension))
        self.groups = groups
        self.eps = 1e-5

    def forward(self, input):
        B, N = input.shape[:2]
        assert N % self.groups == 0

        input_float = input.reshape(B, self.groups, -1, *input.shape[2:]).float()
        input_norm = input_float * torch.rsqrt(input_float.pow(2).mean(2, keepdim=True) + self.eps)
        weight_shape = [1, -1] + [1] * (input.ndim - 2)

        return input_norm.type_as(input).reshape(input.shape) * self.weight.reshape(weight_shape)


class FFN(nn.Module):
    def __init__(self, feature_dim):
        super(FFN, self).__init__()
        self.MLP = nn.Sequential(RMSNorm(feature_dim),
                                 nn.Conv1d(feature_dim, feature_dim * 4, 1, bias=False))
        self.MLP_output = nn.Conv1d(feature_dim * 2, feature_dim, 1, bias=False)

    def forward(self, input):
        gate, z = self.MLP(input).chunk(2, dim=1)
        output = input + self.MLP_output(F.silu(gate) * z)

        return output


class MPEncoderAttention(nn.Module):
    def __init__(self, n_fft, inner_dim, branch_dim, num_head, attention_drop=0.):
        super(MPEncoderAttention, self).__init__()
        assert inner_dim % num_head == 0
        assert branch_dim % num_head == 0
        assert branch_dim * 2 == inner_dim

        self.inner_dim = inner_dim
        self.branch_dim = branch_dim
        self.num_head = num_head
        self.hidden_size = inner_dim // num_head
        self.branch_size = branch_dim // num_head
        self.attention_drop = attention_drop
        self.nband = 80

        self.query = nn.Parameter(torch.randn(self.nband, inner_dim))

        self.input_norm = RMSNorm(inner_dim)
        self.query_norm = RMSNorm(inner_dim)

        self.query_proj = nn.Linear(inner_dim, inner_dim, bias=False)
        self.key_proj = nn.Linear(inner_dim, inner_dim, bias=False)
        self.mag_value_proj = nn.Linear(branch_dim, branch_dim, bias=False)
        self.pha_value_proj = nn.Linear(branch_dim, branch_dim, bias=False)
        self.output = nn.Linear(inner_dim, inner_dim, bias=False)

        self.band_bias = BandBias(n_fft, num_head)
        self.ffn = FFN(inner_dim)

    def forward(self, mag_feature, pha_feature):
        B, L, N = mag_feature.shape
        assert pha_feature.shape == mag_feature.shape
        assert N == self.branch_dim

        routing = torch.cat((mag_feature, pha_feature), dim=-1)
        routing = self.input_norm(routing.transpose(1, 2)).transpose(1, 2).contiguous()

        query = self.query.unsqueeze(0).expand(B, -1, -1)
        query = self.query_norm(query.transpose(1, 2)).transpose(1, 2).contiguous()

        Q = self.query_proj(query)
        Q = Q.reshape(B, self.nband, self.num_head, self.hidden_size)
        Q = Q.transpose(1, 2).contiguous()

        K = self.key_proj(routing)
        K = K.reshape(B, L, self.num_head, self.hidden_size)
        K = K.transpose(1, 2).contiguous()

        mag_value = self.mag_value_proj(mag_feature)
        mag_value = mag_value.reshape(B, L, self.num_head, self.branch_size)
        mag_value = mag_value.transpose(1, 2).contiguous()

        pha_value = self.pha_value_proj(pha_feature)
        pha_value = pha_value.reshape(B, L, self.num_head, self.branch_size)
        pha_value = pha_value.transpose(1, 2).contiguous()

        value = torch.cat((mag_value, pha_value), dim=-1)

        dropout_p = self.attention_drop if self.training else 0.
        output = F.scaled_dot_product_attention(Q,
                                                K,
                                                value,
                                                attn_mask=self.band_bias(dtype=K.dtype),
                                                dropout_p=dropout_p,
                                                is_causal=False)

        output = output.transpose(1, 2).reshape(B, self.nband, self.inner_dim)
        output = self.output(output)
        output = self.ffn(output.transpose(1, 2)).transpose(1, 2).contiguous()

        return output


class MPDecoderAttention(nn.Module):
    def __init__(self, n_fft, inner_dim, branch_dim, num_head, attention_drop=0.):
        super(MPDecoderAttention, self).__init__()
        assert inner_dim % num_head == 0
        assert branch_dim % num_head == 0

        self.enc_dim = n_fft // 2 + 1
        self.inner_dim = inner_dim
        self.branch_dim = branch_dim
        self.num_head = num_head
        self.hidden_size = inner_dim // num_head
        self.branch_size = branch_dim // num_head
        self.attention_drop = attention_drop
        self.nband = 80

        self.mag_query = nn.Parameter(torch.randn(self.enc_dim, inner_dim))
        self.pha_query = nn.Parameter(torch.randn(self.enc_dim, inner_dim))

        self.input_norm = RMSNorm(inner_dim)
        self.mag_query_norm = RMSNorm(inner_dim)
        self.pha_query_norm = RMSNorm(inner_dim)

        self.mag_query_proj = nn.Linear(inner_dim, inner_dim, bias=False)
        self.pha_query_proj = nn.Linear(inner_dim, inner_dim, bias=False)
        self.key_proj = nn.Linear(inner_dim, inner_dim, bias=False)
        self.mag_value_proj = nn.Linear(inner_dim, branch_dim, bias=False)
        self.pha_value_proj = nn.Linear(inner_dim, branch_dim, bias=False)
        self.mag_output = nn.Linear(branch_dim, branch_dim, bias=False)
        self.pha_output = nn.Linear(branch_dim, branch_dim, bias=False)

        self.band_bias = BandBias(n_fft, num_head, decoder=True)
        self.mag_ffn = FFN(branch_dim)
        self.pha_ffn = FFN(branch_dim)

    def forward(self, input):
        B, L, N = input.shape
        assert L == self.nband
        assert N == self.inner_dim

        input = self.input_norm(input.transpose(1, 2)).transpose(1, 2).contiguous()

        mag_query = self.mag_query.unsqueeze(0).expand(B, -1, -1)
        pha_query = self.pha_query.unsqueeze(0).expand(B, -1, -1)

        mag_query = self.mag_query_norm(mag_query.transpose(1, 2))
        mag_query = mag_query.transpose(1, 2).contiguous()

        pha_query = self.pha_query_norm(pha_query.transpose(1, 2))
        pha_query = pha_query.transpose(1, 2).contiguous()

        K = self.key_proj(input)
        K = K.reshape(B, L, self.num_head, self.hidden_size)
        K = K.transpose(1, 2).contiguous()

        mag_query = self.mag_query_proj(mag_query)
        mag_query = mag_query.reshape(B, self.enc_dim, self.num_head, self.hidden_size)
        mag_query = mag_query.transpose(1, 2).contiguous()

        pha_query = self.pha_query_proj(pha_query)
        pha_query = pha_query.reshape(B, self.enc_dim, self.num_head, self.hidden_size)
        pha_query = pha_query.transpose(1, 2).contiguous()

        mag_value = self.mag_value_proj(input)
        mag_value = mag_value.reshape(B, L, self.num_head, self.branch_size)
        mag_value = mag_value.transpose(1, 2).contiguous()

        pha_value = self.pha_value_proj(input)
        pha_value = pha_value.reshape(B, L, self.num_head, self.branch_size)
        pha_value = pha_value.transpose(1, 2).contiguous()

        position_bias = self.band_bias(dtype=K.dtype)
        dropout_p = self.attention_drop if self.training else 0.

        mag_feature = F.scaled_dot_product_attention(mag_query,
                                                     K,
                                                     mag_value,
                                                     attn_mask=position_bias,
                                                     dropout_p=dropout_p,
                                                     is_causal=False)

        pha_feature = F.scaled_dot_product_attention(pha_query,
                                                     K,
                                                     pha_value,
                                                     attn_mask=position_bias,
                                                     dropout_p=dropout_p,
                                                     is_causal=False)

        mag_feature = mag_feature.transpose(1, 2)
        mag_feature = mag_feature.reshape(B, self.enc_dim, self.branch_dim)

        pha_feature = pha_feature.transpose(1, 2)
        pha_feature = pha_feature.reshape(B, self.enc_dim, self.branch_dim)

        mag_feature = self.mag_output(mag_feature)
        pha_feature = self.pha_output(pha_feature)

        mag_feature = self.mag_ffn(mag_feature.transpose(1, 2))
        mag_feature = mag_feature.transpose(1, 2).contiguous()

        pha_feature = self.pha_ffn(pha_feature.transpose(1, 2))
        pha_feature = pha_feature.transpose(1, 2).contiguous()

        return mag_feature, pha_feature


class MPASCEncoder(nn.Module):
    def __init__(self, h):
        super(MPASCEncoder, self).__init__()
        self.n_fft = h.n_fft
        self.enc_dim = h.n_fft // 2 + 1
        self.feature_dim = h.feature_dim
        self.inner_dim = h.asc_inner_dim
        self.branch_dim = h.asc_inner_dim // 2
        self.num_head = h.asc_num_heads
        self.nband = 80

        self.mag_input = nn.Sequential(nn.Conv2d(1, self.branch_dim, 3, padding=1),
                                       RMSNorm(self.branch_dim))

        self.pha_input = nn.Sequential(nn.Conv2d(1, self.branch_dim, 3, padding=1),
                                       RMSNorm(self.branch_dim))

        self.attention = MPEncoderAttention(self.n_fft,
                                            self.inner_dim,
                                            self.branch_dim,
                                            self.num_head,
                                            h.attention_drop)

        self.output = nn.Sequential(nn.Conv2d(self.inner_dim,
                                              self.feature_dim,
                                              3,
                                              padding=1),
                                    RMSNorm(self.feature_dim))

    def forward(self, noisy_mag, noisy_pha):
        assert noisy_mag.shape == noisy_pha.shape
        assert noisy_mag.shape[1] == self.enc_dim

        B, N, T = noisy_mag.shape

        mag_feature = noisy_mag.transpose(1, 2).unsqueeze(1)
        pha_feature = noisy_pha.transpose(1, 2).unsqueeze(1)

        mag_feature = self.mag_input(mag_feature)
        pha_feature = self.pha_input(pha_feature)

        mag_feature = mag_feature.permute(0, 2, 3, 1).contiguous()
        mag_feature = mag_feature.reshape(B * T, N, self.branch_dim)

        pha_feature = pha_feature.permute(0, 2, 3, 1).contiguous()
        pha_feature = pha_feature.reshape(B * T, N, self.branch_dim)

        feature = self.attention(mag_feature, pha_feature)

        feature = feature.reshape(B, T, self.nband, self.inner_dim)
        feature = feature.permute(0, 3, 1, 2).contiguous()
        feature = self.output(feature)
        feature = feature.permute(0, 3, 1, 2).contiguous()

        return feature


class MPASCDecoder(nn.Module):
    def __init__(self, h):
        super(MPASCDecoder, self).__init__()
        self.n_fft = h.n_fft
        self.enc_dim = h.n_fft // 2 + 1
        self.feature_dim = h.feature_dim
        self.inner_dim = h.asc_inner_dim
        self.branch_dim = h.asc_inner_dim // 2
        self.num_head = h.asc_num_heads
        self.nband = 80
        self.eps = 1e-8

        self.input = nn.Sequential(nn.ConvTranspose2d(self.feature_dim,
                                                      self.inner_dim,
                                                      3,
                                                      padding=1),
                                   RMSNorm(self.inner_dim))

        self.attention = MPDecoderAttention(self.n_fft,
                                            self.inner_dim,
                                            self.branch_dim,
                                            self.num_head,
                                            h.attention_drop)

        self.mag_output = nn.Sequential(RMSNorm(self.branch_dim),
                                        nn.Conv2d(self.branch_dim, 2, 3, padding=1),
                                        nn.GLU(dim=1))

        self.pha_output = nn.Sequential(RMSNorm(self.branch_dim),
                                        nn.Conv2d(self.branch_dim, 4, 3, padding=1),
                                        nn.GLU(dim=1))

    def forward(self, feature, noisy_mag, noisy_pha):
        assert feature.shape[1] == self.nband
        assert feature.shape[2] == self.feature_dim
        assert noisy_mag.shape == noisy_pha.shape
        assert noisy_mag.shape[1] == self.enc_dim

        B, nband, N, T = feature.shape

        feature = feature.permute(0, 2, 3, 1).contiguous()
        feature = self.input(feature)
        feature = feature.permute(0, 2, 3, 1).contiguous()
        feature = feature.reshape(B * T, nband, self.inner_dim)

        mag_feature, pha_feature = self.attention(feature)

        mag_feature = mag_feature.reshape(B, T, self.enc_dim, self.branch_dim)
        mag_feature = mag_feature.permute(0, 3, 1, 2).contiguous()

        pha_feature = pha_feature.reshape(B, T, self.enc_dim, self.branch_dim)
        pha_feature = pha_feature.permute(0, 3, 1, 2).contiguous()

        delta_mag = self.mag_output(mag_feature)
        delta_mag = delta_mag.squeeze(1).transpose(1, 2).contiguous()
        pred_mag = torch.clamp(noisy_mag + delta_mag, min=self.eps)

        phase_vec = self.pha_output(pha_feature)
        pred_pha = torch.atan2(phase_vec[:, 1], phase_vec[:, 0])
        pred_pha = pred_pha.transpose(1, 2).contiguous()

        pred_com = torch.stack((pred_mag * torch.cos(pred_pha),
                                pred_mag * torch.sin(pred_pha)),
                               dim=-1)

        return pred_mag, pred_pha, pred_com