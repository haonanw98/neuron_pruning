from functools import partial
import os
import pathlib
import shutil
import math

import torch
import torch.nn as nn
import torch.tensor as tensor

from args import args as parser_args
import pdb

def print_global_layerwise_prune_rate(model, prune_rate):
    score_threshold = get_global_score_threshold(model, prune_rate)
    print("rank_method: ",parser_args.rank_method,";whether_abs: ",parser_args.whether_abs)
    print("score_threshold: ",score_threshold)
    for n, m in model.named_modules():
        if hasattr(m, "scores") and m.prune_rate != 0:
            shape = m.scores.shape
            if parser_args.pmode == "normal":
                scores = m.scores.abs().flatten()
                pruned_num = scores[scores <= score_threshold].size()[0]
                total_num = scores.shape[0]
                print(n, " pruned: ", pruned_num, " total: ", total_num, " rate: ", pruned_num / total_num)
            elif parser_args.pmode == "channel":
                channel_size = shape[1] * shape[2] * shape[3]
                if parser_args.rank_method == "absolute":
                    scores = m.scores.abs().sum((1, 2, 3)).flatten() / channel_size
                elif parser_args.rank_method == "relevant":
                    if parser_args.whether_abs == "abs":
                        scores = torch.div(m.scores.abs().sum((1,2,3)).flatten(),m.sumofabsofinit.cuda())
                    else:
                        scores = torch.div(m.scores.sum((1,2,3)).flatten(),m.sumofabsofinit.cuda())
                #print(scores)
                pruned_num = scores[scores < score_threshold].size()[0] * channel_size
                total_num = scores.shape[0] * channel_size
                print(n, " pruned: ", pruned_num, " total: ", total_num, " rate: ", pruned_num / total_num)
        else:
            print(n)

def print_model_scores(model):
    for n, m in model.named_modules():
        if hasattr(m, "scores"):
            print(n, "scores:", m.scores)

def get_global_score_threshold(model, prune_rate):
    all_scores = None
    if prune_rate == 0:
        # YHT modification, since I delete abs, 0 no longer make sense
        return -10000
    # YHT modification 
    for n, m in model.named_modules():
        if hasattr(m, "scores") and m.prune_rate != 0:
            shape = m.scores.shape
            if all_scores is None:
                all_scores = tensor([]).to(m.scores.device)
            if parser_args.rank_method == "absolute":
                if parser_args.pmode == "normal":
                    all_scores = torch.cat([all_scores, m.scores.abs().flatten()])
                elif parser_args.pmode == "channel":
                    channel_size = shape[1] * shape[2] * shape[3]
                    all_scores = torch.cat([all_scores, m.scores.abs().sum((1, 2, 3)).flatten() / channel_size])
            elif parser_args.rank_method == "relevant":
                assert parser_args.pmode == "channel","only channel pmode could use relevant method!"
                channel_size = shape[1] * shape[2] * shape[3]
                # noabs / abs of init
                if parser_args.whether_abs == "abs":
                    attach = torch.div(m.scores.abs().sum((1,2,3)).flatten(), m.sumofabsofinit.cuda())
                else:
                    attach = torch.div(m.scores.sum((1,2,3)).flatten(), m.sumofabsofinit.cuda())
                all_scores = torch.cat([all_scores,attach])
            else:
                print("wrong rank_method! Only absolute and relevant is supported.")
                raise
    return torch.kthvalue(all_scores, int(prune_rate * all_scores.numel())).values.item()
    # End of modification
   
def save_checkpoint(state, is_best, filename="checkpoint.pth", save=False):
    filename = pathlib.Path(filename)

    if not filename.parent.exists():
        os.makedirs(filename.parent)

    torch.save(state, filename)

    if is_best:
        shutil.copyfile(filename, str(filename.parent / "model_best.pth"))

        if not save:
            os.remove(filename)


def get_lr(optimizer):
    return optimizer.param_groups[0]["lr"]


def freeze_model_weights(model):
    print("=> Freezing model weights")

    for n, m in model.named_modules():
        if hasattr(m, "weight") and m.weight is not None:
            print(f"==> No gradient to {n}.weight")
            m.weight.requires_grad = False
            if m.weight.grad is not None:
                print(f"==> Setting gradient of {n}.weight to None")
                m.weight.grad = None

            if hasattr(m, "bias") and m.bias is not None:
                print(f"==> No gradient to {n}.bias")
                m.bias.requires_grad = False

                if m.bias.grad is not None:
                    print(f"==> Setting gradient of {n}.bias to None")
                    m.bias.grad = None


def freeze_model_subnet(model):
    print("=> Freezing model subnet")

    for n, m in model.named_modules():
        if hasattr(m, "scores"):
            m.scores.requires_grad = False
            print(f"==> No gradient to {n}.scores")
            if m.scores.grad is not None:
                print(f"==> Setting gradient of {n}.scores to None")
                m.scores.grad = None


def unfreeze_model_weights(model):
    print("=> Unfreezing model weights")

    for n, m in model.named_modules():
        if hasattr(m, "weight") and m.weight is not None:
            print(f"==> Gradient to {n}.weight")
            m.weight.requires_grad = True
            if hasattr(m, "bias") and m.bias is not None:
                print(f"==> Gradient to {n}.bias")
                m.bias.requires_grad = True


def unfreeze_model_subnet(model):
    print("=> Unfreezing model subnet")

    for n, m in model.named_modules():
        if hasattr(m, "scores"):
            print(f"==> Gradient to {n}.scores")
            m.scores.requires_grad = True


def set_model_prune_rate(model, prune_rate):
    print(f"==> Setting prune rate of network to {prune_rate}")
    ind = -1
    for n, m in model.named_modules():
        if hasattr(m, "set_prune_rate"):
            ind += 1
            if parser_args.protect is not None:
                if parser_args.protect == "linear":
                    if 'linear' in n:
                        print(f"==> Setting prune rate of {n} to 0")
                        m.set_prune_rate(0)
                        continue
                elif parser_args.protect == "linear_last":
                    if 'linear' in n and n.split('.')[1] == str(len(model.linear) - 1):
                        print(f"==> Setting prune rate of {n} to 0")
                        m.set_prune_rate(0)
                        continue
                else:
                    raise(ValueError)

            if not parser_args.prandom: 
                m.set_prune_rate(prune_rate)
                print(f"==> Setting prune rate of {n} to {prune_rate}")
            else:
                layer_prune_rate = prune_rate
                if ind < len(parser_args.prlist):
                    layer_prune_rate = parser_args.prlist[ind]
                    print("WARNING: prune rate list length might not be correct")
                m.set_prune_rate(layer_prune_rate)
                print(f"==> Setting prune rate of {n} to {layer_prune_rate}")

def accumulate(model, f):
    acc = 0.0

    for child in model.children():
        acc += accumulate(child, f)

    acc += f(model)

    return acc


class LabelSmoothing(nn.Module):
    """
    NLL loss with label smoothing.
    """

    def __init__(self, smoothing=0.0):
        """
        Constructor for the LabelSmoothing module.

        :param smoothing: label smoothing factor
        """
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)

        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


class SubnetL1RegLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, model, temperature=1.0):
        l1_accum = 0.0
        for n, p in model.named_parameters():
            if n.endswith("scores"):
                l1_accum += (p*temperature).sigmoid().sum()

        return l1_accum


