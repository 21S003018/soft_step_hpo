from trainers import CNNTrainer, EvalTrainer, NasTrainer, SoftStepTrainer
from const import *
import torch


def softstep_eval(order=1, dataset=CIFAR10):
    config_path = "log/softstep_linear_1e-6/{}_o{}_{}.json"
    if order == 1:
        # linear,o1,cifar10, according to w train loss
        indexs = [129, 235, 305, 229]
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
    '''case evaluation'''
#    trainer = EvalTrainer(
#        CIFAR100, "log/softstep_linear_1e-5/194_o1_cifar-100-python.json")
#    trainer.train()

    '''mobilenetv2 evaluation'''
    # trainer = CNNTrainer(MOBILENET, CIFAR10)
    # trainer.train()
    # trainer = CNNTrainer(MOBILENET, CIFAR100)
    # trainer.train()
    '''linear searchspace evaluation'''
    # trainer = EvalTrainer(CIFAR10, "config/search_space_eval.json")
    # trainer.train()
    # trainer = EvalTrainer(CIFAR100, "config/search_space_eval.json")
    # trainer.train()
    '''softstep search linear searchspace'''
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
    #     SOFTSTEP, CIFAR100, path=LINEARSEARCHSPACE, opt_order=2)
    # trainer.train()
    '''softstep(linear,o1,cifar10) evaluation'''
    # softstep_eval(1, CIFAR10)
    '''softstep(linear,o2,cifar10) evaluation'''
    # softstep_eval(2, CIFAR10)
    '''softstep(linear,o1,cifar100) evaluation'''
    # softstep_eval(1, CIFAR100)
    '''softstep(linear,o2,cifar100) evaluation'''
    # softstep_eval(2, CIFAR100)

    '''bottleneck searchspace evaluation'''
    '''softstep search bottleneck searchspace'''
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
    '''softstep(bottleneck,o1,cifar10) evaluation'''
    # softstep_eval(1, CIFAR10)
    '''softstep(bottleneck,o2,cifar10) evaluation'''
    # softstep_eval(2, CIFAR10)
    pass
