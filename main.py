from trainers import CNNTrainer, NasTrainer
from const import *


if __name__ == "__main__":
    # cnn model
    # trainer = CNNTrainer(RESNET, CIFAR10)
    #trainer = CNNTrainer(MOBILENET, CIFAR10)
    # trainer = CNNTrainer(SHUFFLENET, CIFAR10)
    trainer = NasTrainer(SOFTSTEP, CIFAR10, path=RESIDUALCIFAR10)
    # nas model
    trainer.train()
    print(trainer.get_metrics())
