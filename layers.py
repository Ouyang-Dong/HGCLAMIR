
import torch
import torch.autograd
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Attention(nn.Module):
    def __init__(self, temperature, attn_dropout=0.1):

        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
        # softmax+dropout
        attn = attn / abs(attn.min())
        attn = self.dropout(F.softmax(F.normalize(attn, dim=-1), dim=-1))
        output = torch.matmul(attn, v)

        return output, attn, v


class FeedForwardLayer(nn.Module):

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()

        self.w_1 = nn.Linear(d_in, d_hid)
        self.w_2 = nn.Linear(d_hid, d_in)
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x

        x = self.w_2(F.gelu(self.w_1(x)))
        x = self.dropout(x)
        x += residual

        x = self.layer_norm(x)

        return x


class VariLengthInputLayer(nn.Module):
    def __init__(self, input_data_dims, d_k, d_v, n_head, dropout):
        super(VariLengthInputLayer, self).__init__()
        self.n_head = n_head
        self.dims = input_data_dims
        self.d_k = d_k
        self.d_v = d_v
        self.w_qs = []
        self.w_ks = []
        self.w_vs = []
        for i, dim in enumerate(self.dims):
            self.w_q = nn.Linear(dim, n_head * d_k, bias=False) 
            self.w_k = nn.Linear(dim, n_head * d_k, bias=False) 
            self.w_v = nn.Linear(dim, n_head * d_v, bias=False) 

            self.w_qs.append(self.w_q)
            self.w_ks.append(self.w_k)
            self.w_vs.append(self.w_v)
            self.add_module('linear_q_%d_%d' % (dim, i), self.w_q)
            self.add_module('linear_k_%d_%d' % (dim, i), self.w_k)
            self.add_module('linear_v_%d_%d' % (dim, i), self.w_v)

        self.attention = Attention(temperature=d_k ** 0.5, attn_dropout=dropout)
        self.fc = nn.Linear(n_head * d_v, n_head * d_v)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(n_head * d_v, eps=1e-6)

    def forward(self, input_data, mask=None):

        temp_dim = 0
        bs = input_data.size(0)
        modal_num = len(self.dims)
        q = torch.zeros(bs, modal_num, self.n_head * self.d_k).to(device)
        k = torch.zeros(bs, modal_num, self.n_head * self.d_k).to(device)
        v = torch.zeros(bs, modal_num, self.n_head * self.d_v).to(device)

        for i in range(modal_num):
            w_q = self.w_qs[i]
            w_k = self.w_ks[i]
            w_v = self.w_vs[i]

            data = input_data[:, temp_dim: temp_dim + self.dims[i]]
            temp_dim += self.dims[i]
            q[:, i, :] = w_q(data)
            k[:, i, :] = w_k(data)
            v[:, i, :] = w_v(data)

        q = q.view(bs, modal_num, self.n_head, self.d_k)
        k = k.view(bs, modal_num, self.n_head, self.d_k)
        v = v.view(bs, modal_num, self.n_head, self.d_v)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        q, attn, residual = self.attention(q, k, v)
        q = q.transpose(1, 2).contiguous().view(bs, modal_num, -1)
        residual = residual.transpose(1, 2).contiguous().view(bs, modal_num, -1)
        q = self.dropout(self.fc(q))
        q += residual
        q = self.layer_norm(q)

        return q, attn


class EncodeLayer(nn.Module):
    def __init__(self, d_model, d_k, d_v, n_head, dropout):
        super(EncodeLayer, self).__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_q = nn.Linear(d_model, n_head * d_k, bias=False) 
        self.w_k = nn.Linear(d_model, n_head * d_k, bias=False) 
        self.w_v = nn.Linear(d_model, n_head * d_v, bias=False) 

        self.fc = nn.Linear(n_head * d_v, d_model, bias=False) 
        self.attention = Attention(temperature=d_k ** 0.5)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, q, k, v, modal_num, mask=None):
        bs = q.size(0)
        residual = q
        q = self.w_q(q).view(bs, modal_num, self.n_head, self.d_k)
        k = self.w_k(k).view(bs, modal_num, self.n_head, self.d_k)
        v = self.w_v(v).view(bs, modal_num, self.n_head, self.d_v)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        q, attn, _ = self.attention(q, k, v, mask=mask)
        q = q.transpose(1, 2).contiguous().view(bs, modal_num, -1)
        q = self.dropout(self.fc(q))
        q += residual
        q = self.layer_norm(q)
        return q, attn


