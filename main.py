from trainers import CNNTrainer
from const import *


if __name__ == "__main__":
    #trainer = CNNTrainer(RESNET, CIFAR10)
    #trainer = CNNTrainer(MOBILENET, CIFAR10)
    trainer = CNNTrainer(SHUFFLENET, CIFAR10)
    trainer.train()
    print(trainer.get_metrics())
