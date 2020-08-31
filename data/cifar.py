import os
import torch
import torchvision
from torchvision import transforms
import random
from torch.utils.data.sampler import SubsetRandomSampler
from args import args
import numpy as np

class CIFAR10:
    def __init__(self, args):
        super(CIFAR10, self).__init__()
        data_root = os.path.join(args.data, "cifar10")

        use_cuda = torch.cuda.is_available()

        # Data loading code
        kwargs = {"num_workers": args.workers, "pin_memory": True} if use_cuda else {}

        normalize = transforms.Normalize(
            mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262]
        )

        train_dataset = torchvision.datasets.CIFAR10(
            root=data_root,
            train=True,
            download=True,
            transform=transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ]
            ),
        )

        # shuffle training data (by yty)
        if args.shuffle:
            labels = np.array(train_dataset.targets)
            #YHT change
            np.random.seed(args.seed)
            labels = np.random.shuffle(labels)
            train_dataset.targers = labels;
            print("shuffling labels")

        self.train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs
        )

        test_dataset = torchvision.datasets.CIFAR10(
            root=data_root,
            train=False,
            download=True,
            transform=transforms.Compose([transforms.ToTensor(), normalize]),
        )

        # shuffle test data (by yty)
        if args.shuffle:
            labels = np.array(test_dataset.targets)
            #YHT change
            np.random.seed(args.seed)
            labels = np.random.shuffle(labels)
            test_dataset.targers = labels;

        self.val_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=args.batch_size, shuffle=False, **kwargs
        )