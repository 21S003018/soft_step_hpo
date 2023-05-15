from const import *
import sklearn.preprocessing as sp
from sklearn.model_selection import train_test_split
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch
import math
import re
import os
import pickle
import numpy as np
import pandas as pd


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
                                 batch_size=32, shuffle=True,
                                 )
        return train_loader, test_loader, 3, 32, 100

    def load_cinic(self):
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CINICMEAN, CINICSTD),
        ])
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CINICMEAN, CINICSTD),
        ])
        data_root_path = "data/cinic-10/"
        train_dataset = datasets.ImageFolder(
            os.path.join(data_root_path, "train"), train_transform)
        valid_dataset = datasets.ImageFolder(
            os.path.join(data_root_path, "valid"), transform)
        test_dataset = datasets.ImageFolder(
            os.path.join(data_root_path, "test"), transform)

        train_loader = DataLoader(
            dataset=train_dataset, batch_size=64, shuffle=True, num_workers=4,)
        valid_loader = DataLoader(dataset=valid_dataset,
                                  batch_size=100, shuffle=True, num_workers=4,)
        test_loader = DataLoader(dataset=test_dataset,
                                 batch_size=100, shuffle=True, num_workers=4,)
        return train_loader, test_loader, 3, 32, 10

    def load_food(self):
        train_transform = transforms.Compose([
            transforms.RandomRotation(30),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.AutoAugment(),
            transforms.ToTensor(),
            transforms.Normalize(FOODMEAN, FOODSTD),
        ])
        test_transform = transforms.Compose([
            transforms.Resize(255),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(FOODMEAN, FOODSTD),
        ])
        data_root_path = "data/food-101/"
        train_dataset = datasets.ImageFolder(
            os.path.join(data_root_path, "train"), train_transform)
        test_dataset = datasets.ImageFolder(
            os.path.join(data_root_path, "test"), test_transform)

        train_loader = DataLoader(
            dataset=train_dataset, batch_size=64, shuffle=True, num_workers=4,)
        test_loader = DataLoader(dataset=test_dataset,
                                 batch_size=100, shuffle=True, num_workers=4,)
        return train_loader, test_loader, 3, 224, 101

    def load_iris(self):
        LabelIndex = 4
        path = "data/iris/iris.data"
        df = pd.read_csv(path, header=None)
        dataset = np.column_stack((df.values[:, :-1],
                                  sp.LabelEncoder().fit_transform(
            df[[LabelIndex]].values)))
        dataset = np.array(dataset, dtype=float)
        dataset = torch.Tensor(dataset)
        if torch.cuda.is_available():
            dataset = dataset.cuda()
        x_train, x_test, y_train, y_test = train_test_split(
            dataset[:, :-1], dataset[:, -1:].reshape(len(dataset)), test_size=0.2, random_state=0)
        return 4, 3, (x_train, y_train), (x_test, y_test)

    def load_wine(self):
        LabelIndex = 0
        path = "data/wine/wine.data"
        df = pd.read_csv(path, header=None)
        dataset = np.column_stack((df.values[:, 1:],
                                   sp.LabelEncoder().fit_transform(df[[LabelIndex]].values)))
        dataset = np.array(dataset, dtype=float)
        dataset = torch.Tensor(dataset)
        if torch.cuda.is_available():
            dataset = dataset.cuda()
        x_train, x_test, y_train, y_test = train_test_split(
            dataset[:, :-1], dataset[:, -1:].reshape(len(dataset)), test_size=0.2, random_state=0)
        return 13, 3, (x_train, y_train), (x_test, y_test)

    def load_car(self):
        LabelIndex = 6
        path = "data/car/car.data"
        df = pd.read_csv(path, header=None)
        dataset = np.column_stack((sp.OneHotEncoder(sparse=False).fit_transform(df.values[:, :-1]),
                                   sp.LabelEncoder().fit_transform(df[[LabelIndex]].values)))
        dataset = np.array(dataset, dtype=float)
        dataset = torch.Tensor(dataset)
        if torch.cuda.is_available():
            dataset = dataset.cuda()
        x_train, x_test, y_train, y_test = train_test_split(
            dataset[:, :-1], dataset[:, -1:].reshape(len(dataset)), test_size=0.2, random_state=0)
        return 21, 4, (x_train, y_train), (x_test, y_test)

    def load_agaricus_lepiota(self):
        LabelIndex = 0
        path = "data/agaricus-lepiota/agaricus-lepiota.data"
        df = pd.read_csv(path, header=None)
        dataset = np.column_stack((sp.OneHotEncoder(sparse=False).fit_transform(df.values[:, 1:11]),
                                   sp.OneHotEncoder(sparse=False).fit_transform(
                                       df.values[:, 12:]),
                                   sp.LabelEncoder().fit_transform(df[[LabelIndex]].values)))
        dataset = np.array(dataset, dtype=float)
        dataset = torch.Tensor(dataset)
        if torch.cuda.is_available():
            dataset = dataset.cuda()
        x_train, x_test, y_train, y_test = train_test_split(
            dataset[:, :-1], dataset[:, -1:].reshape(len(dataset)), test_size=0.2, random_state=0)
        return 112, 2, (x_train, y_train), (x_test, y_test)

    def get(self, dataset):
        if dataset == CIFAR10:
            return self.load_cifar10()
        if dataset == CIFAR100:
            return self.load_cifar100()
        if dataset == IRIS:
            return self.load_iris()
        if dataset == WINE:
            return self.load_wine()
        if dataset == CAR:
            return self.load_car()
        if dataset == AGARICUS:
            return self.load_agaricus_lepiota()
        if dataset == CINIC:
            return self.load_cinic()
        if dataset == FOOD:
            return self.load_food()
        return None


def newton_expansion(c):
    def f(x):
        return 2*c*x-2-(math.exp(x/2) + math.exp(-x/2))

    def f_derive(x):
        return 2*c - (1/2*math.exp(x/2)-1/2*math.exp(-x/2))
    x = 2*math.log(4*c)*1.1
    for i in range(10):
        x = x - f(x)/f_derive(x)
    return x


def num_image(loader):
    res = 0
    for _, label in loader:
        res += len(label)
    return res


if __name__ == "__main__":
    train_loader, test_loader, _, _, _ = Data().get(FOOD)
    # for img, label in train_loader:
    #     print(img.size())
    print(len(train_loader))
    pass
