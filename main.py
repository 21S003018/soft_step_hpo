from trainers import CNNTrainer, NasTrainer
from const import *
import torch


if __name__ == "__main__":
    trainer = CNNTrainer(RESNET, CIFAR100)
    trainer = CNNTrainer(MOBILENET, CIFAR100)
    trainer = NasTrainer(SOFTSTEP, CIFAR100, path=SEARCHSPACE)
    # trainer.train()
    preds = trainer.model(torch.randn((1,3,32,32)))
    print(preds.size())
    pass