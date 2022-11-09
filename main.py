from trainers import CNNTrainer, NasTrainer, SoftStepTrainer
from const import *
import torch


if __name__ == "__main__":
    # trainer = CNNTrainer(MOBILENET, CIFAR100)
    # trainer.train()
    # trainer = CNNTrainer(MOBILENET, CIFAR10)
    # trainer.train()

    # trainer = CNNTrainer(MOBILENET, CIFAR100)
    trainer = SoftStepTrainer(SOFTSTEP, CIFAR10, path=SEARCHSPACE)
    trainer.train()
    # x = torch.randn((1, 3, 32, 32))
    # x = x.cuda(DEVICE)
    # preds = trainer.model(x)
    # # print(preds.size())
    pass
