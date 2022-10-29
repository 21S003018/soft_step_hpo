from const import *
import sklearn.preprocessing as sp
from sklearn.model_selection import train_test_split
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import torch
import pickle


class Data():
    def __init__(self) -> None:
        self.datasets = DATASETS
        if torch.cuda.is_available():
            self.device = "cuda:3"
        else:
            self.device = "cpu"
        return

    def load_cifar10(self):
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10MEAN, CIFAR10STD),
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10MEAN, CIFAR10STD),
        ])
        data_root_path = "data/"
        train_dataset = datasets.CIFAR10(root=data_root_path, train=True,
                                         transform=train_transform,
                                         download=True)
        test_dataset = datasets.CIFAR10(root=data_root_path, train=False,
                                        transform=test_transform, download=True)
        train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=NAME2BATCHSIZE[CIFAR10], shuffle=True,
                                  num_workers=4,
                                  )
        test_loader = DataLoader(dataset=test_dataset,
                                 batch_size=BATCHSIZE, shuffle=True,
                                 )
        return train_loader, test_loader, 3, 32, 10

    def load_cifar100(self):
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100MEAN, CIFAR100STD),
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100MEAN, CIFAR100STD),
        ])
        data_root_path = "data/"
        train_dataset = datasets.CIFAR100(root=data_root_path, train=True,
                                          transform=train_transform, download=True)
        test_dataset = datasets.CIFAR100(root=data_root_path, train=False,
                                         transform=test_transform, download=True)
        train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=NAME2BATCHSIZE[CIFAR100], shuffle=True,
                                  num_workers=4,
                                  )
        test_loader = DataLoader(dataset=test_dataset,
                                 batch_size=BATCHSIZE, shuffle=True,
                                 )
        return train_loader, test_loader, 3, 32, 100

    def load_mnist(self):
        train_transform = transforms.Compose([
            transforms.RandomCrop(28, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(MNISTMEAN, MNISTSTD),
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(MNISTMEAN, MNISTSTD),
        ])
        data_root_path = "data/"
        train_dataset = datasets.MNIST(root=data_root_path, train=True,
                                       transform=train_transform, download=True)
        test_dataset = datasets.MNIST(root=data_root_path, train=False,
                                      transform=test_transform, download=True)
        train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=NAME2BATCHSIZE[MNIST], shuffle=True,
                                  num_workers=4,
                                  )
        test_loader = DataLoader(dataset=test_dataset,
                                 batch_size=BATCHSIZE, shuffle=True,
                                 )
        return train_loader, test_loader, 1, 28, 10

    def load_svhn(self):
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(SVHNMEAN, SVHNSTD),
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(SVHNMEAN, SVHNSTD),
        ])
        data_root_path = "data/SVHN/"
        train_dataset = datasets.SVHN(root=data_root_path, split="train",
                                      transform=train_transform, download=True)
        test_dataset = datasets.SVHN(root=data_root_path, split="test",
                                     transform=test_transform, download=True)
        train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=NAME2BATCHSIZE[SVHN], shuffle=True,
                                  num_workers=4)
        test_loader = DataLoader(dataset=test_dataset,
                                 batch_size=BATCHSIZE, shuffle=True)
        return train_loader, test_loader, 3, 32, 10

    def get(self, dataset):
        if dataset == MNIST:
            return self.load_mnist()
        if dataset == SVHN:
            return self.load_svhn()
        if dataset == CIFAR10:
            return self.load_cifar10()
        if dataset == CIFAR100:
            return self.load_cifar100()
        return None


def num_image(loader):
    res = 0
    for _, label in loader:
        res += len(label)
    return res
