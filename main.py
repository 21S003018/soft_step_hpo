from trainers import CNNTrainer, EvalTrainer, NasTrainer, SoftStepTrainer
from const import *
import torch


if __name__ == "__main__":
    # trainer = CNNTrainer(MOBILENET, CIFAR100)
    # trainer.train()
    # trainer = CNNTrainer(MOBILENET, CIFAR10)
    # trainer.train()
    # trainer = EvalTrainer(CIFAR10, "config/search_space_linear.json")
    # trainer.train(save=True, epochs=int(EPOCHS/2))

    # trainer = SoftStepTrainer(
    #     SOFTSTEP, CIFAR10, path=LINEARSEARCHSPACE, opt_order=1)
    # trainer.train()
    # trainer = SoftStepTrainer(
    #     SOFTSTEP, CIFAR10, path=LINEARSEARCHSPACE, opt_order=2)
    # trainer.train()
    trainer = SoftStepTrainer(
        SOFTSTEP, CIFAR100, path=LINEARSEARCHSPACE, opt_order=1)
    trainer.train()
    # trainer = SoftStepTrainer(
    #     SOFTSTEP, CIFAR10, path=LINEARSEARCHSPACE, opt_order=2)
    # trainer.train()
    pass
