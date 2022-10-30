from trainers import CNNTrainer, NasTrainer, SoftStepTrainer
from const import *
import torch


if __name__ == "__main__":
    # trainer = CNNTrainer(RESNET, CIFAR100)
    # trainer = CNNTrainer(MOBILENET, CIFAR100)
    trainer = SoftStepTrainer(SOFTSTEP, CIFAR100, path=SEARCHSPACE)
    # trainer.train()
    x = torch.randn((1, 3, 64, 64))
    x = x.cuda(DEVICE)
    preds = trainer.model(x)
    # print(preds.size())
    pass
