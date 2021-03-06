# device
DEVICE = "cuda:3"
# datasets
CIFAR10 = "cifar-10-batches-py"
CIFAR100 = "cifar-100-python"
MNIST = "mnist"
SVHN = "svhn"
DATASETS = [CIFAR10, CIFAR100, MNIST, SVHN]
NUMIMAGE = {
    MNIST: 60000,
    SVHN: 73257,
    CIFAR10: 50000,
    CIFAR100: 50000
}
CIFAR10MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10STD = (0.2470, 0.2435, 0.2616)
CIFAR100MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
CIFAR100STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
MNISTMEAN = (0.1307,)
MNISTSTD = (0.3081,)
SVHNMEAN = (0.4376821219921112, 0.4437697231769562, 0.4728044271469116)
SVHNSTD = (0.19803012907505035, 0.2010156363248825, 0.19703614711761475)


# """hyper-params"""
P_MOMENTUM = 0.9
BATCHSIZE = 64
EPOCHS = 200
EXPANDTIMES = 20
NAME2BATCHSIZE = {
    CIFAR10: 128,
    CIFAR100: 32,
    MNIST: 128,
    SVHN: 64,
}
# cnn model name
RESNET = "ResNet"
MOBILENET = "mobilenetv2"
SHUFFLENET = "shufflenet"
SOFTSTEP = "SoftStep"

# struc config
RESIDUALCIFAR10 = "config/residual_cifar10.json"
INVERTEDRESIDUALIMAGENET = "config/inverted_residual_imagenet.json"

if __name__ == "__main__":
    pass
