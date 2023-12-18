import random
import os
import numpy as np
import torch
import math
from torch import nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from layers import VariLengthInputLayer, EncodeLayer, FeedForwardLayer


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

seed_torch(seed=1234)


class TransformerEncoder(nn.Module):
    def __init__(self, input_data_dims, hyperpm):
        super(TransformerEncoder, self).__init__()
        self.hyperpm = hyperpm
        self.input_data_dims = input_data_dims
        self.d_q = hyperpm.n_hidden
        self.d_k = hyperpm.n_hidden
        self.d_v = hyperpm.n_hidden
        self.n_head = hyperpm.n_head
        self.dropout = hyperpm.dropout
        self.n_layer = hyperpm.nlayer
        self.modal_num = hyperpm.nmodal
        self.d_out = self.d_v * self.n_head * self.modal_num

        self.InputLayer = VariLengthInputLayer(self.input_data_dims, self.d_k, self.d_v, self.n_head, self.dropout)

        self.Encoder = []
        self.FeedForward = []

        for i in range(self.n_layer):
            encoder = EncodeLayer(self.d_k * self.n_head, self.d_k, self.d_v, self.n_head, self.dropout)
            self.add_module('encode_%d' % i, encoder)
            self.Encoder.append(encoder)

            feedforward = FeedForwardLayer(self.d_v * self.n_head, self.d_v * self.n_head, dropout=self.dropout)
            self.add_module('feed_%d' % i, feedforward)
            self.FeedForward.append(feedforward)


    def forward(self, x):
        bs = x.size(0)
        attn_map = []
        x, _attn = self.InputLayer(x)

        attn = _attn.mean(dim=1)
        attn_map.append(attn.detach().cpu().numpy())

        for i in range(self.n_layer):
            x, _attn = self.Encoder[i](q=x, k=x, v=x, modal_num=self.modal_num)
            attn = _attn.mean(dim=1)
            x = self.FeedForward[i](x)
            attn_map.append(attn.detach().cpu().numpy())

        x = x.view(bs, -1)

        # output = self.Outputlayer(x)
        return x


class HGNN_conv(nn.Module):
    def __init__(self, in_ft, out_ft, bias=True):
        super(HGNN_conv, self).__init__()
        self.in_features = in_ft
        self.out_features = out_ft
        self.weight = Parameter(torch.Tensor(in_ft, out_ft))
        if bias:
            self.bias = Parameter(torch.Tensor(out_ft))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, G):
        x = x.matmul(self.weight)
        if self.bias is not None:
            x = x + self.bias
        x = G.matmul(x)
        return x

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class HGCN(nn.Module):
    def __init__(self, in_dim, hidden_list, dropout = 0.5):
        super(HGCN, self).__init__()
        self.dropout = dropout

        self.hgnn1 = HGNN_conv(in_dim, hidden_list[0])

    def forward(self,x, G):

        x_embed = self.hgnn1(x, G)
        x_embed_1 = F.leaky_relu(x_embed, 0.25)


        return x_embed_1

class CL_HGCN(nn.Module):
    def __init__(self, in_size, hid_list, num_proj_hidden, alpha = 0.5):
        super(CL_HGCN, self).__init__()
        self.hgcn1 = HGCN(in_size, hid_list)
        self.hgcn2 = HGCN(in_size, hid_list)

        self.fc1 = torch.nn.Linear(hid_list[-1], num_proj_hidden)
        self.fc2 = torch.nn.Linear(num_proj_hidden, hid_list[-1])

        self.tau = 0.5
        self.alpha = alpha

    def forward(self, x1, adj1, x2, adj2):

        z1 = self.hgcn1(x1, adj1)
        h1 = self.projection(z1)

        z2 = self.hgcn2(x2, adj2)
        h2 = self.projection(z2)

        loss = self.alpha*self.sim(h1, h2) + (1-self.alpha)*self.sim(h2,h1)

        return z1, z2, loss

    def projection(self, z):
        z = F.elu(self.fc1(z))
        return self.fc2(z)

    def norm_sim(self, z1, z2):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def sim(self, z1, z2):
        f = lambda x: torch.exp(x / self.tau)
        refl_sim = f(self.norm_sim(z1, z1))
        between_sim = f(self.norm_sim(z1, z2))
        loss = -torch.log(between_sim.diag() / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))
        loss = loss.sum(dim=-1).mean()
        return loss


class HGCN_Attention_mechanism(nn.Module):
    def __init__(self):
        super(HGCN_Attention_mechanism,self).__init__()
        self.hiddim = 64

        self.fc_x1 = nn.Linear(in_features=2, out_features=self.hiddim)
        self.fc_x2 = nn.Linear(in_features=self.hiddim, out_features=2)
        self.sigmoidx = nn.Sigmoid()


    def forward(self,input_list):

        XM = torch.cat((input_list[0], input_list[1]), 1).t()
        XM = XM.view(1, 1 * 2, input_list[0].shape[1], -1)

        globalAvgPool_x = nn.AvgPool2d((input_list[0].shape[1], input_list[0].shape[0]), (1, 1))
        x_channel_attenttion = globalAvgPool_x(XM)

        x_channel_attenttion = x_channel_attenttion.view(x_channel_attenttion.size(0), -1)
        x_channel_attenttion = self.fc_x1(x_channel_attenttion)
        x_channel_attenttion = torch.relu(x_channel_attenttion)
        x_channel_attenttion = self.fc_x2(x_channel_attenttion)
        x_channel_attenttion = self.sigmoidx(x_channel_attenttion)
        x_channel_attenttion = x_channel_attenttion.view(x_channel_attenttion.size(0), x_channel_attenttion.size(1), 1, 1)

        XM_channel_attention = x_channel_attenttion * XM
        XM_channel_attention = torch.relu(XM_channel_attention)

        return XM_channel_attention[0]


class HGCLAMIR(nn.Module):
    def __init__(self, mi_num, dis_num, hidd_list, num_proj_hidden, hyperpm):
        super(HGCLAMIR, self).__init__()

        self.CL_HGCN_mi = CL_HGCN(mi_num + dis_num, hidd_list,num_proj_hidden)
        self.CL_HGCN_dis = CL_HGCN(dis_num + mi_num, hidd_list,num_proj_hidden)


        self.AM_mi = HGCN_Attention_mechanism()
        self.AM_dis = HGCN_Attention_mechanism()

        self.Transformer_mi = TransformerEncoder([hidd_list[-1],hidd_list[-1]], hyperpm)
        self.Transformer_dis = TransformerEncoder([hidd_list[-1],hidd_list[-1]], hyperpm)

      
        self.linear_x_1 = nn.Linear(hyperpm.n_head*hyperpm.n_hidden*hyperpm.nmodal, 256)
        self.linear_x_2 = nn.Linear(256, 128)
        self.linear_x_3 = nn.Linear(128, 64)

        self.linear_y_1 = nn.Linear(hyperpm.n_head*hyperpm.n_hidden*hyperpm.nmodal, 256)
        self.linear_y_2 = nn.Linear(256, 128)
        self.linear_y_3 = nn.Linear(128, 64)


    def forward(self, concat_mi_tensor, concat_dis_tensor, G_mi_Kn, G_mi_Km, G_dis_Kn, G_dis_Km):

        mi_embedded = concat_mi_tensor
        dis_embedded = concat_dis_tensor

        
        mi_feature1, mi_feature2, mi_cl_loss = self.CL_HGCN_mi(mi_embedded, G_mi_Kn, mi_embedded, G_mi_Km)
        mi_feature_att = self.AM_mi([mi_feature1,mi_feature2])
        mi_feature_att1 = mi_feature_att[0].t()
        mi_feature_att2 = mi_feature_att[1].t()
        mi_concat_feature = torch.cat([mi_feature_att1, mi_feature_att2], dim=1)
        mi_feature = self.Transformer_mi(mi_concat_feature)

       
        dis_feature1, dis_feature2, dis_cl_loss = self.CL_HGCN_dis(dis_embedded, G_dis_Kn, dis_embedded, G_dis_Km)
        dis_feature_att = self.AM_dis([dis_feature1,dis_feature2])
        dis_feature_att1 = dis_feature_att[0].t()
        dis_feature_att2 = dis_feature_att[1].t()
        dis_concat_feature = torch.cat([dis_feature_att1, dis_feature_att2], dim=1)
        dis_feature = self.Transformer_dis(dis_concat_feature)

        x1 = torch.relu(self.linear_x_1(mi_feature))
        x2 = torch.relu(self.linear_x_2(x1))
        x = torch.relu(self.linear_x_3(x2))

        y1 = torch.relu(self.linear_y_1(dis_feature))
        y2 = torch.relu(self.linear_y_2(y1))
        y = torch.relu(self.linear_y_3(y2))

        
        score = x.mm(y.t())

        return score, mi_cl_loss, dis_cl_loss



