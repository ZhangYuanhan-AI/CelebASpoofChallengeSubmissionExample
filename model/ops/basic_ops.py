import torch
import math


class Identity(torch.nn.Module):
    def forward(self, input):
        return input


class SegmentConsensus(torch.autograd.Function):

    # def __init__(self, consensus_type, dim=1):
    #     self.consensus_type = consensus_type
    #     self.dim = dim
    #     self.shape = None

    @staticmethod
    def forward(ctx, input_tensor, consensus_tensor, dim_tensor):
        # shape = input_tensor.size()
        if consensus_tensor == 0:
            output = input_tensor.mean(dim=dim_tensor.item(), keepdim=True)
        elif consensus_tensor == 1:
            output = input_tensor
        else:
            output = None
        ctx.save_for_backward(input_tensor, consensus_tensor, dim_tensor) ## save input_tensor

        return output

    @staticmethod
    def backward(ctx, grad_output):
        input_tensor, consensus_tensor, dim_tensor = ctx.saved_tensors  ## get input_tensor.shape
        shape = input_tensor.size()
        if consensus_tensor == 0:
            grad_in = grad_output.expand(shape) / float(shape[dim_tensor.item()])
        elif consensus_tensor == 1:
            grad_in = grad_output
        else:
            grad_in = None

        return grad_in


class ConsensusModule(torch.nn.Module):

    def __init__(self, consensus_type, dim=1):
        super(ConsensusModule, self).__init__()
        self.consensus_type = consensus_type if consensus_type != 'rnn' else 'identity'
        self.dim = dim

    def forward(self, input):
        if self.consensus_type == 'avg':
            consensus_tensor = torch.tensor(0).cuda()
        elif self.consensus_type == 'identity':
            consensus_tensor = torch.tensor(1).cuda()
        else:
            consensus_tensor = torch.tensor(2).cuda()
        dim_tensor = torch.tensor(self.dim).cuda()
        f = SegmentConsensus.apply
        return f(input, consensus_tensor, dim_tensor) ## str, int -> tensor
