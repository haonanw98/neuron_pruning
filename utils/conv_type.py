import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import pdb
import math

from args import args as parser_args


DenseConv = nn.Conv2d


class GetSubnet(autograd.Function):
    @staticmethod
    def forward(ctx, noabs_scores, k, sumofinit):
        # Get the subnetwork by sorting the scores and using the top k%
        scores = noabs_scores.abs()
        # flat_out and out access the same memory.
        out = scores.clone()
        shape = scores.shape
        # WHN modification
        pmode, pscale, score_threshold = parser_args.pmode, parser_args.pscale, parser_args.score_threshold
        if pmode == "normal" and pscale == "layerwise":
            _, idx = scores.flatten().sort()
            j = int(k * scores.numel())
            flat_out = out.flatten()
            flat_out[idx[:j]] = 0
            flat_out[idx[j:]] = 1

        elif pmode == "channel" and pscale == "layerwise":
            channel_num = shape[0]
            channel_size = shape[1] * shape[2] * shape[3]
            # get sum for each channel
            _, idx = scores.sum((1, 2, 3)).flatten().sort()
            j = int(k * scores.sum((1, 2, 3)).numel())
            flat_out = out.flatten().reshape(channel_num, -1)
            flat_out[idx[:j]] = 0
            flat_out[idx[j:]] = 1

        elif pmode == "normal" and pscale == "global":
            flat_out = out.flatten()
            idx = scores.flatten() > score_threshold
            flat_out[flat_out != 0] = 0
            flat_out[idx] = 1

        elif pmode == "channel" and pscale == "global":
            channel_num = shape[0]
            channel_size = shape[1] * shape[2] * shape[3]
            # YHT modification
            if parser_args.rank_method == "absolute":
                idx = (scores.sum((1, 2, 3)).flatten() /
                       channel_size) > score_threshold
            elif parser_args.rank_method == "relevant":
                if parser_args.whether_abs == "abs":
                    idx = torch.div(scores.sum((1, 2, 3)).flatten(),
                                    sumofinit.cuda()) >= score_threshold
                else:
                    idx = torch.div(noabs_scores.sum(
                        (1, 2, 3)).flatten(), sumofinit.cuda()) >= score_threshold
            # End of modification
            flat_out = out.flatten().reshape(channel_num, -1)
            flat_out[flat_out != 0] = 0
            flat_out[idx] = 1

        else:
            print("Unexpected pruning type.")
            raise
        # End of WHN modification
        return out

    @staticmethod
    def backward(ctx, g):
        # send the gradient g straight-through on the backward pass.
        return g, None, None

# Not learning weights, finding subnet


class SubnetConv(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.scores = nn.Parameter(torch.Tensor(self.weight.size()))
        # YHT modification
        if parser_args.score_init == "kaiming":
            nn.init.kaiming_uniform_(self.scores, a=math.sqrt(5))
        elif parser_args.score_init == "normal":
            nn.init.uniform_(self.scores)
        elif parser_args.score_init == "gaussian":
            nn.init.normal_(self.scores)
        else:
            print("unkown type of score_init!")
            raise
        # adding anotehr init_score which is used to compare RELEVANT scaling
        self.init_scores = nn.Parameter(self.scores, requires_grad=False)
        self.sumofabsofinit = self.init_scores.abs().sum((1, 2, 3)).flatten()
        # End of modification

    def set_prune_rate(self, prune_rate):
        self.prune_rate = prune_rate

    @property
    def clamped_scores(self):
        return self.scores.abs()

    # YHT modification
    @property
    def clamped_sumofinit(self):
        return self.sumofabsofinit
    # End of modification

    def forward(self, x):
        # YHT modification
        subnet = GetSubnet.apply(
            self.scores, self.prune_rate, self.clamped_sumofinit)
        # YHT modification
        w = self.weight * subnet
        x = F.conv2d(
            x, w, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        return x


"""
Sample Based Sparsification
"""


class StraightThroughBinomialSample(autograd.Function):
    @staticmethod
    def forward(ctx, scores):
        output = (torch.rand_like(scores) < scores).float()
        return output

    @staticmethod
    def backward(ctx, grad_outputs):
        return grad_outputs, None


class BinomialSample(autograd.Function):
    @staticmethod
    def forward(ctx, scores):
        output = (torch.rand_like(scores) < scores).float()
        ctx.save_for_backward(output)

        return output

    @staticmethod
    def backward(ctx, grad_outputs):
        subnet, = ctx.saved_variables

        grad_inputs = grad_outputs.clone()
        grad_inputs[subnet == 0.0] = 0.0

        return grad_inputs, None


# Not learning weights, finding subnet
class SampleSubnetConv(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.scores = nn.Parameter(torch.Tensor(self.weight.size()))
        if parser_args.score_init_constant is not None:
            self.scores.data = (
                torch.ones_like(self.scores) * parser_args.score_init_constant
            )
        else:
            nn.init.kaiming_uniform_(self.scores, a=math.sqrt(5))

    @property
    def clamped_scores(self):
        return torch.sigmoid(self.scores)

    def forward(self, x):
        subnet = StraightThroughBinomialSample.apply(self.clamped_scores)
        w = self.weight * subnet
        x = F.conv2d(
            x, w, self.bias, self.stride, self.padding, self.dilation, self.groups
        )

        return x


"""
Fixed subnets 
"""


class FixedSubnetConv(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.scores = nn.Parameter(torch.Tensor(self.weight.size()))
        nn.init.kaiming_uniform_(self.scores, a=math.sqrt(5))

    def set_prune_rate(self, prune_rate):
        self.prune_rate = prune_rate
        print("prune_rate_{}".format(self.prune_rate))

    def set_subnet(self):
        output = self.clamped_scores().clone()
        _, idx = self.clamped_scores().flatten().abs().sort()
        p = int(self.prune_rate * self.clamped_scores().numel())
        flat_oup = output.flatten()
        flat_oup[idx[:p]] = 0
        flat_oup[idx[p:]] = 1
        self.scores = torch.nn.Parameter(output)
        self.scores.requires_grad = False

    def clamped_scores(self):
        return self.scores.abs()

    def get_subnet(self):
        return self.weight * self.scores

    def forward(self, x):
        w = self.get_subnet()
        x = F.conv2d(
            x, w, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        return x
