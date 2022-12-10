from trainers import CNNTrainer, EvalTrainer, NasTrainer, SoftStepTrainer, HPOTrainer
from const import *
import torch
import argparse

# parser = argparse.ArgumentParser()

# parser.add_argument('--cuda', type=str, default='0',help='')

# device = f"cuda:{parser.cuda}"


if __name__ == "__main__":
    '''case evaluation'''
    trainer = HPOTrainer(policy_name="rand", dataset=CIFAR10,
                         search_space=LINEARSEARCHSPACE, device='cuda:0')
    # trainer.pre_train()
    trainer.train()
    '''mobilenetv2 evaluation'''
    # trainer = CNNTrainer(MOBILENET, CIFAR10,device="cuda:0")
    # trainer.train()
    # trainer = CNNTrainer(MOBILENET, CIFAR100)
    # trainer.train()
    '''resnet evaluation'''
    # trainer = CNNTrainer(RESNET, CIFAR10, device="cuda:3")
    # trainer.train()
    # trainer = CNNTrainer(RESNET, CIFAR100, device="cuda:2")
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
    #     SOFTSTEP, CIFAR100, path=BOTTLENECKSEARCHSPACE, opt_order=1)
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
    #     SOFTSTEP, CIFAR100, path=BOTTLENECKSEARCHSPACE, opt_order=1, device="cuda:0")
    # trainer.train()
    '''softstep(bottleneck,o1,cifar10) evaluation'''
    # softstep_eval(1, CIFAR10)
    '''softstep(bottleneck,o2,cifar10) evaluation'''
    # softstep_eval(2, CIFAR10)

    '''softstep search shallow searchspace'''
    # trainer = SoftStepTrainer(
    #     SOFTSTEP, CIFAR10, path=SHALLOWSEARCHSPACE, opt_order=1)
    # trainer.train()
    # trainer = SoftStepTrainer(
    #     SOFTSTEP, CIFAR100, path=SHALLOWSEARCHSPACE, opt_order=1,device="cuda:0")
    # trainer.train()
    pass
