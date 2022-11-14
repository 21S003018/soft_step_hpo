from trainers import CNNTrainer, EvalTrainer, TEvalTrainer, NasTrainer, SoftStepTrainer
from const import *
import torch


def softstep_eval(order=1, dataset=CIFAR10):
    config_path = "log/softstep/{}_o{}_{}.json"
    if order == 1:
        # linear,o1,cifar10, according to w train loss
        indexs = [323, 357, 367, 315, 391]
        # indexs = [316, 324, 330, 332, 386, 392]  # linear,o1,cifar10, according to accu
    elif order == 2:
        indexs = [396, 394, 330, 376, 339]  # linear,o2,cifar10
    for index in indexs:
        path = config_path.format(index, order, dataset)
        print("current evaluate: ", path)
        trainer = EvalTrainer(dataset, path)
        trainer.train()
    return


if __name__ == "__main__":
    # trainer = CNNTrainer(MOBILENET, CIFAR100)
    # trainer.train()
    # trainer = CNNTrainer(MOBILENET, CIFAR10)
    # trainer.train()
    '''softstep(linear,o1,cifar10) evaluation '''
    # softstep_eval(1, CIFAR10)
    '''softstep(linear,o2,cifar10) evaluation '''
    # softstep_eval(2, CIFAR10)
    '''linear search space evaluation'''
    trainer = TEvalTrainer(CIFAR10, "config/search_space_eval.json")
    trainer.train()
    # trainer = EvalTrainer(CIFAR100, "config/search_space_linear.json")
    # trainer.train(save=True)

    '''softstep search inverted residual space'''
    # trainer = SoftStepTrainer(
    #     SOFTSTEP, CIFAR10, path=LINEARSEARCHSPACE, opt_order=1)
    # trainer.train()
    # trainer = SoftStepTrainer(
    #     SOFTSTEP, CIFAR10, path=LINEARSEARCHSPACE, opt_order=2)
    # trainer.train()
    # trainer = SoftStepTrainer(
    #     SOFTSTEP, CIFAR100, path=LINEARSEARCHSPACE, opt_order=1)
    # trainer.train()
    # trainer = SoftStepTrainer(
    #     SOFTSTEP, CIFAR100, path=LINEARSEARCHSPACE, opt_order=2)
    # trainer.train()
    '''softstep search residual space'''
    # trainer = SoftStepTrainer(
    #     SOFTSTEP, CIFAR10, path=BOTTLENECKSEARCHSPACE, opt_order=1)
    # trainer.train()
    # trainer = SoftStepTrainer(
    #     SOFTSTEP, CIFAR10, path=BOTTLENECKSEARCHSPACE, opt_order=2)
    # trainer.train()
    # trainer = SoftStepTrainer(
    #     SOFTSTEP, CIFAR100, path=BOTTLENECKSEARCHSPACE, opt_order=1)
    # trainer.train()
    # trainer = SoftStepTrainer(
    #     SOFTSTEP, CIFAR100, path=BOTTLENECKSEARCHSPACE, opt_order=2)
    # trainer.train()
    pass
