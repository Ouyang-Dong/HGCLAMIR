from torch import nn
from param import parameter_parser
args = parameter_parser()


class Myloss(nn.Module):
    def __init__(self):
        super(Myloss, self).__init__()

    def forward(self, one_index, zero_index, input, target):
        loss = nn.MSELoss(reduction='none')
        loss_sum = loss(input, target)
        
        return (1-args.alpha)*loss_sum[one_index].sum() + args.alpha*loss_sum[zero_index].sum()


def get_L2reg(parameters):
    reg = 0
    for param in parameters:
        reg += 0.5 * (param ** 2).sum()
    return reg
