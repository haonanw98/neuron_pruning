import os
import pathlib
import random
import time

from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import pdb

from utils.conv_type import FixedSubnetConv, SampleSubnetConv
from utils.logging import AverageMeter, ProgressMeter
from utils.net_utils import (
    set_model_prune_rate,
    freeze_model_weights,
    # SJT modification:
    freeze_model_subnet,
    unfreeze_model_weights,
    unfreeze_model_subnet,
    # End of SJT modification
    save_checkpoint,
    get_lr,
    LabelSmoothing,
    get_global_score_threshold,  # WHN modification
    print_global_layerwise_prune_rate, # WHN modification
)
from utils.schedulers import get_policy


from args import args
import importlib

import data
import models


def main():
    print(args)
    # YHT modification
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    else:
        if args.shuffle:
            print("Using shuffle mode without specify argd.seed!")
            return
    
    # End of YHT modification
    
    # Simply call main_worker function
    main_worker(args)


def main_worker(args):
    args.gpu = None
    train, validate, modifier = get_trainer(args)
    
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    # create model and optimizer
    model = get_model(args)
    model = set_gpu(args, model)

    if args.pretrained:
        pretrained(args, model)
    
    # SJT modification:
    if args.exp_mode: # pretraining/pruning/funetuning
        exp_mode = args.exp_mode
        if exp_mode == "pretraining":
            # YHT modefication, setting the pruning rate to 0
            print("Figure out your exp_mode is pretraining, setting prune-rate to 0")
            args.prune_rate = 0
            unfreeze_model_weights(model)
            freeze_model_subnet(model)
    # End of SJT modification    
        
        
    optimizer = get_optimizer(args, model)
    data = get_dataset(args)
    lr_policy = get_policy(args.lr_policy)(optimizer, args)

    if args.label_smoothing is None:
        criterion = nn.CrossEntropyLoss().cuda()
    else:
        criterion = LabelSmoothing(smoothing=args.label_smoothing)

    # optionally resume from a checkpoint
    best_acc1 = 0.0
    best_acc5 = 0.0
    best_train_acc1 = 0.0
    best_train_acc5 = 0.0


    if args.resume:
        # SJT modification
        if args.exp_mode:
            if args.exp_mode == "pruning":
                optimizer = resume_pruning(args,model)
            else:# Only can be "finetuning"
                if args.exp_mode != "finetuning":
                    print("resume method should be combined with pruning/finetuning exp_mode together!")
                    return
                else:
                    optimizer = resume_finetuning(args,model)
                    print("#####################DEBUG PRINT : VALIDATE FIRST#####################")
                    validate(data.val_loader, model, criterion, args, writer= None, epoch=args.start_epoch)
        else:
            best_acc1 = resume(args,model,optimizer)
        # End of SJT modification

    # Data loading code
    if args.evaluate:
        acc1, acc5 = validate(
            data.val_loader, model, criterion, args, writer=None, epoch=args.start_epoch
        )

        return

    # Set up directories
    run_base_dir, ckpt_base_dir, log_base_dir = get_directories(args)
    args.ckpt_base_dir = ckpt_base_dir

    writer = SummaryWriter(log_dir=log_base_dir)
    epoch_time = AverageMeter("epoch_time", ":.4f", write_avg=False)
    validation_time = AverageMeter("validation_time", ":.4f", write_avg=False)
    train_time = AverageMeter("train_time", ":.4f", write_avg=False)
    progress_overall = ProgressMeter(
        1, [epoch_time, validation_time, train_time], prefix="Overall Timing"
    )

    end_epoch = time.time()
    args.start_epoch = args.start_epoch or 0
    acc1 = None         
        
    
    
    # Save the initial state
    save_checkpoint(
        {
            "epoch": 0,
            "arch": args.arch,
            "state_dict": model.state_dict(),
            "best_acc1": best_acc1,
            "best_acc5": best_acc5,
            "best_train_acc1": best_train_acc1,
            "best_train_acc5": best_train_acc5,
            "optimizer": optimizer.state_dict(),
            "curr_acc1": acc1 if acc1 else "Not evaluated",
        },
        False,
        filename=ckpt_base_dir / f"initial.state",
        save=False,
    )

    if args.gp_warm_up:
        record_prune_rate = args.prune_rate
    if args.print_more:
        print_global_layerwise_prune_rate(model,args.prune_rate)
    # Start training
    for epoch in range(args.start_epoch, args.epochs):
        lr_policy(epoch, iteration=None)
        modifier(args, epoch, model)

        cur_lr = get_lr(optimizer)

        # train for one epoch
        start_train = time.time()
        # WHN modeification add global pruning
        if args.pscale == "global":
            if args.gp_warm_up:
                if epoch < args.gp_warm_up_epochs:
                    args.prune_rate = 0
                else:
                    args.prune_rate = record_prune_rate
            args.score_threshold = get_global_score_threshold(model, args.prune_rate)
        
        train_acc1, train_acc5 = train(
            data.train_loader, model, criterion, optimizer, epoch, args, writer=writer
        )
        train_time.update((time.time() - start_train) / 60)
        
        # YHT modification
        if args.print_more:
            print_global_layerwise_prune_rate(model,args.prune_rate)
        # End of modification

        # evaluate on validation set
        start_validation = time.time()
        # if random labeled, evaluate on training set (by yty)
        if args.shuffle:
            acc1, acc5 = train_acc1, train_acc5
        else:
            acc1, acc5 = validate(data.val_loader, model, criterion, args, writer, epoch)

        validation_time.update((time.time() - start_validation) / 60)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)
        best_acc5 = max(acc5, best_acc5)
        best_train_acc1 = max(train_acc1, best_train_acc1)
        best_train_acc5 = max(train_acc5, best_train_acc5)

        save = ((epoch % args.save_every) == 0) and args.save_every > 0
        if is_best or save or epoch == args.epochs - 1:
            if is_best:
                print(f"==> New best, saving at {ckpt_base_dir / 'model_best.pth'}")

            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "arch": args.arch,
                    "state_dict": model.state_dict(),
                    "best_acc1": best_acc1,
                    "best_acc5": best_acc5,
                    "best_train_acc1": best_train_acc1,
                    "best_train_acc5": best_train_acc5,
                    "optimizer": optimizer.state_dict(),
                    "curr_acc1": acc1,
                    "curr_acc5": acc5,
                },
                is_best,
                filename=ckpt_base_dir / f"epoch_{epoch}.state",
                save=save,
            )

        epoch_time.update((time.time() - end_epoch) / 60)
        progress_overall.display(epoch)
        progress_overall.write_to_tensorboard(
            writer, prefix="diagnostics", global_step=epoch
        )

        if args.conv_type == "SampleSubnetConv":
            count = 0
            sum_pr = 0.0
            for n, m in model.named_modules():
                if isinstance(m, SampleSubnetConv):
                    # avg pr across 10 samples
                    pr = 0.0
                    for _ in range(10):
                        pr += (
                            (torch.rand_like(m.clamped_scores) >= m.clamped_scores)
                            .float()
                            .mean()
                            .item()
                        )
                    pr /= 10.0
                    writer.add_scalar("pr/{}".format(n), pr, epoch)
                    sum_pr += pr
                    count += 1

            args.prune_rate = sum_pr / count
            writer.add_scalar("pr/average", args.prune_rate, epoch)

        writer.add_scalar("test/lr", cur_lr, epoch)
        end_epoch = time.time()

    write_result_to_csv(
        best_acc1=best_acc1,
        best_acc5=best_acc5,
        best_train_acc1=best_train_acc1,
        best_train_acc5=best_train_acc5,
        prune_rate=args.prune_rate,
        curr_acc1=acc1,
        curr_acc5=acc5,
        base_config=args.config,
        name=args.name,
    )


def get_trainer(args):
    print(f"=> Using trainer from trainers.{args.trainer}")
    trainer = importlib.import_module(f"trainers.{args.trainer}")

    return trainer.train, trainer.validate, trainer.modifier


def set_gpu(args, model):
    assert torch.cuda.is_available(), "CPU-only experiments currently unsupported"

    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    elif args.multigpu is None:
        device = torch.device("cpu")
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        print(f"=> Parallelizing on {args.multigpu} gpus")
        torch.cuda.set_device(args.multigpu[0])
        args.gpu = args.multigpu[0]
        model = torch.nn.DataParallel(model, device_ids=args.multigpu).cuda(
            args.multigpu[0]
        )

    cudnn.benchmark = True

    return model


def resume(args, model, optimizer):
    if os.path.isfile(args.resume):
        print(f"=> Loading checkpoint '{args.resume}'")

        checkpoint = torch.load(args.resume, map_location=f"cuda:{args.multigpu[0]}")
        if args.start_epoch is None:
            print(f"=> Setting new start epoch at {checkpoint['epoch']}")
            args.start_epoch = checkpoint["epoch"]

        best_acc1 = checkpoint["best_acc1"]

        model.load_state_dict(checkpoint["state_dict"])

        optimizer.load_state_dict(checkpoint["optimizer"])

        print(f"=> Loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']})")

        return best_acc1
    else:
        print(f"=> No checkpoint found at '{args.resume}'")
       

# SJT modification:
def resume_pruning(args,model):
    if os.path.isfile(args.resume):
        print("############################ WELCOME to the PRUNING MODE!! ############################")
        # Load the pretrained model
        print(f"=> Loading checkpoint '{args.resume}'")
        checkpoint = torch.load(args.resume, map_location=f"cuda:{args.multigpu[0]}")    
        model.load_state_dict(checkpoint["state_dict"]) # Read the weights & scores, use the prune-rate in the config file, i.e. 0
        
        # Freeze the weights & unfreeze the scores
        freeze_model_weights(model)
        unfreeze_model_subnet(model)
        
        # Set the pruning rate
        set_model_prune_rate(model, prune_rate=args.prune_rate)# Need to specify the prune_rate
        
        # Reset the Optimizer
        print("Under the pruning setting, the weights/scores' gradient(s) are as follows:")
        print("You should see that weights are FROZEN and scores are UNFROZEN")
        print("############################ DEBUG PRINT ##########################")
        
        #get_optimizer(args,model)
        print(
            f"=> Rough estimate model params {sum(int(p.numel() * (1-args.prune_rate)) for n, p in model.named_parameters() if not n.endswith('scores'))}"
        )
        
        # Reset the start_epoch and epochs, we use the DEFAULT epochs = 100 this quantity was modified at line76 & line260
        args.start_epoch = 0 # this quantity was modified at line76 & line260, we use the DEFAULT epochs = 20
        
        # Reset the exp-name
        args.name = args.name + "_pruning" # Change the exp-name so that the new model can be saved to a new directory
        model = set_gpu(args,model)
    else:
        print(f"=> No checkpoint found at '{args.resume}'")
    return get_optimizer(args,model)
        
        
def resume_finetuning(args,model):
    if os.path.isfile(args.resume):
        print("############################ WELCOME to the FINETUNING MODE!! ############################")
        # Load the pretrained model
        print(f"=> Loading checkpoint '{args.resume}'")
        checkpoint = torch.load(args.resume, map_location=f"cuda:{args.multigpu[0]}")    
        model.load_state_dict(checkpoint["state_dict"])
        # Freeze the weights & unfreeze the scores
        unfreeze_model_weights(model)
        freeze_model_subnet(model)
        
        # Set the pruning rate
        set_model_prune_rate(model, prune_rate=args.prune_rate) # Don't need to specify the prune_rate
        
        # Reset the Optimizer
        print("Under the pruning setting, the weights/scores' gradient(s) are as follows:")
        print("You should see that weights are UNFROZEN and scores are FROZEN")
        print("############################ DEBUG PRINT ############################")
        
        #get_optimizer(args,model)
        print(
            f"=> Rough estimate model params {sum(int(p.numel() * (1-args.prune_rate)) for n, p in model.named_parameters() if not n.endswith('scores'))}"
        )
        print("############################ FINETUNING: EPOCHS = 20 ############################")
        # Reset the start_epoch and epochs, we use the DEFAULT epochs = 20 this quantity was modified at line76 & line260
        args.start_epoch = 0 # this quantity was modified at line76 & line260, we use the DEFAULT epochs = 20
        
        # Reset the exp-name
        args.name = args.name + "_finetuning" # Change the exp-name so that the new model can be saved to a new directory
        model = set_gpu(args,model)
    else:
        print(f"=> No checkpoint found at '{args.resume}'")
    return get_optimizer(args,model)
# End of SJT modification

def pretrained(args, model):
    if os.path.isfile(args.pretrained):
        print("=> loading pretrained weights from '{}'".format(args.pretrained))
        pretrained = torch.load(
            args.pretrained,
            map_location=torch.device("cuda:{}".format(args.multigpu[0])),
        )["state_dict"]

        model_state_dict = model.state_dict()
        for k, v in pretrained.items():
            if k not in model_state_dict or v.size() != model_state_dict[k].size():
                print("IGNORE:", k)
        pretrained = {
            k: v
            for k, v in pretrained.items()
            if (k in model_state_dict and v.size() == model_state_dict[k].size())
        }
        model_state_dict.update(pretrained)
        model.load_state_dict(model_state_dict)

    else:
        print("=> no pretrained weights found at '{}'".format(args.pretrained))

    for n, m in model.named_modules():
        if isinstance(m, FixedSubnetConv):
            m.set_subnet()


def get_dataset(args):
    print(f"=> Getting {args.set} dataset")
    dataset = getattr(data, args.set)(args)

    return dataset


def get_model(args):
    if args.first_layer_dense:
        args.first_layer_type = "DenseConv"

    print("=> Creating model '{}'".format(args.arch))
    model = models.__dict__[args.arch]()
    
    # applying sparsity to the network
    if (
        args.conv_type != "DenseConv"
        and args.conv_type != "SampleSubnetConv"
        and args.conv_type != "ContinuousSparseConv"
    ):
        ### YHT modification in May 7
        if args.prune_rate < 0 and args.exp_mode == "pruning":
            raise ValueError("Need to set a positive prune rate")

        set_model_prune_rate(model, prune_rate=args.prune_rate)
        print(
            f"=> Rough estimate model params {sum(int(p.numel() * (1-args.prune_rate)) for n, p in model.named_parameters() if not n.endswith('scores'))}"
        )

    # freezing the weights if we are only doing subnet training
    if args.freeze_weights:
        freeze_model_weights(model)

    return model


def get_optimizer(args, model):
    for n, v in model.named_parameters():
        if v.requires_grad:
            print("<DEBUG> gradient to", n)

        if not v.requires_grad:
            print("<DEBUG> no gradient to", n)

    if args.optimizer == "sgd":
        parameters = list(model.named_parameters())
        bn_params = [v for n, v in parameters if ("bn" in n) and v.requires_grad]
        rest_params = [v for n, v in parameters if ("bn" not in n) and v.requires_grad]
        optimizer = torch.optim.SGD(
            [
                {
                    "params": bn_params,
                    "weight_decay": 0 if args.no_bn_decay else args.weight_decay,
                },
                {"params": rest_params, "weight_decay": args.weight_decay},
            ],
            args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov=args.nesterov,
        )
    elif args.optimizer == "adam":
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr
        )

    return optimizer


def _run_dir_exists(run_base_dir):
    log_base_dir = run_base_dir / "logs"
    ckpt_base_dir = run_base_dir / "checkpoints"

    return log_base_dir.exists() or ckpt_base_dir.exists()


def get_directories(args):
    if args.config is None or args.name is None:
        raise ValueError("Must have name and config")

    config = pathlib.Path(args.config).stem
    if args.log_dir is None:
        run_base_dir = pathlib.Path(
            f"runs/{config}/{args.name}/prune_rate={args.prune_rate}"
        )
    else:
        run_base_dir = pathlib.Path(
            f"{args.log_dir}/{config}/{args.name}/prune_rate={args.prune_rate}"
        )
    if args.width_mult != 1.0:
        run_base_dir = run_base_dir / "width_mult={}".format(str(args.width_mult))

    if _run_dir_exists(run_base_dir):
        rep_count = 0
        while _run_dir_exists(run_base_dir / str(rep_count)):
            rep_count += 1

        run_base_dir = run_base_dir / str(rep_count)

    log_base_dir = run_base_dir / "logs"
    ckpt_base_dir = run_base_dir / "checkpoints"

    if not run_base_dir.exists():
        os.makedirs(run_base_dir)

    (run_base_dir / "settings.txt").write_text(str(args))

    return run_base_dir, ckpt_base_dir, log_base_dir


def write_result_to_csv(**kwargs):
    results = pathlib.Path("runs") / "results.csv"

    if not results.exists():
        results.write_text(
            "Date Finished, "
            "Base Config, "
            "Name, "
            "Prune Rate, "
            "Current Val Top 1, "
            "Current Val Top 5, "
            "Best Val Top 1, "
            "Best Val Top 5, "
            "Best Train Top 1, "
            "Best Train Top 5\n"
        )

    now = time.strftime("%m-%d-%y_%H:%M:%S")

    with open(results, "a+") as f:
        f.write(
            (
                "{now}, "
                "{base_config}, "
                "{name}, "
                "{prune_rate}, "
                "{curr_acc1:.02f}, "
                "{curr_acc5:.02f}, "
                "{best_acc1:.02f}, "
                "{best_acc5:.02f}, "
                "{best_train_acc1:.02f}, "
                "{best_train_acc5:.02f}\n"
            ).format(now=now, **kwargs)
        )


if __name__ == "__main__":
    main()
