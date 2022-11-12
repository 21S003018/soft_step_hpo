from trainers import CNNTrainer, EvalTrainer, NasTrainer, SoftStepTrainer
from const import *
import torch


if __name__ == "__main__":
    # trainer = CNNTrainer(MOBILENET, CIFAR100)
    # trainer.train()
    # trainer = CNNTrainer(MOBILENET, CIFAR10)
    # trainer.train()
    # trainer = EvalTrainer(CIFAR10, "config/softstep_linear_cifar10_o1_l1.json")
    # trainer.train()

    trainer = SoftStepTrainer(
        SOFTSTEP, CIFAR10, path=LINEARSEARCHSPACE, opt_order=1)
    trainer.train()
    trainer = SoftStepTrainer(
        SOFTSTEP, CIFAR10, path=LINEARSEARCHSPACE, opt_order=2)
    trainer.train()
    pass
