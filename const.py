# device
DEVICE = "cuda:0"
# datasets
CIFAR10 = "cifar-10-batches-py"
CIFAR100 = "cifar-100-python"
MNIST = "mnist"
SVHN = "svhn"
CINIC = "cinic"
FOOD = "food"
IRIS = "iris"
WINE = "wine"
CAR = "car"
AGARICUS = "agaricus_lepiota"

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
CINICMEAN = [0.47889522, 0.47227842, 0.43047404]
CINICSTD = [0.24205776, 0.23828046, 0.25874835]
FOODMEAN = [0.485, 0.456, 0.406]
FOODSTD = [0.229, 0.224, 0.225]

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
MOBILENET = "MobileNetV2"
# nas model name
SOFTSTEP = "SoftStep"

# struc config
SHALLOWSEARCHSPACE = "config/search_space_shallow.json"
BOTTLENECKSEARCHSPACE = "config/search_space_bottleneck.json"
LINEARSEARCHSPACE = "config/search_space_linear.json"
SEARCHSPACE = "config/search_space.json"

# hparams bound for linear search space
conv_in_channel = "conv_in_channel"
block_in_conv1_channel = "block_in_conv1_channel"
block_in_conv2_kernel = "block_in_conv2_kernel"
block_in_conv3_channel = "block_in_conv3_channel"
stages_0_block_conv1_channel = "stages_0_block_conv1_channel"
stages_0_block_conv2_kernel = "stages_0_block_conv2_kernel"
stages_0_block_conv3_channel = "stages_0_block_conv3_channel"
stages_0_skips_0_conv1_channel = "stages_0_skips_0_conv1_channel"
stages_0_skips_0_conv2_kernel = "stages_0_skips_0_conv2_kernel"
stages_0_skips_1_conv1_channel = "stages_0_skips_1_conv1_channel"
stages_0_skips_1_conv2_kernel = "stages_0_skips_1_conv2_kernel"
stages_1_block_conv1_channel = "stages_1_block_conv1_channel"
stages_1_block_conv2_kernel = "stages_1_block_conv2_kernel"
stages_1_block_conv3_channel = "stages_1_block_conv3_channel"
stages_1_skips_0_conv1_channel = "stages_1_skips_0_conv1_channel"
stages_1_skips_0_conv2_kernel = "stages_1_skips_0_conv2_kernel"
stages_1_skips_1_conv1_channel = "stages_1_skips_1_conv1_channel"
stages_1_skips_1_conv2_kernel = "stages_1_skips_1_conv2_kernel"
stages_2_block_conv1_channel = "stages_2_block_conv1_channel"
stages_2_block_conv2_kernel = "stages_2_block_conv2_kernel"
stages_2_block_conv3_channel = "stages_2_block_conv3_channel"
stages_2_skips_0_conv1_channel = "stages_2_skips_0_conv1_channel"
stages_2_skips_0_conv2_kernel = "stages_2_skips_0_conv2_kernel"
stages_2_skips_1_conv1_channel = "stages_2_skips_1_conv1_channel"
stages_2_skips_1_conv2_kernel = "stages_2_skips_1_conv2_kernel"
stages_3_block_conv1_channel = "stages_3_block_conv1_channel"
stages_3_block_conv2_kernel = "stages_3_block_conv2_kernel"
stages_3_block_conv3_channel = "stages_3_block_conv3_channel"
stages_3_skips_0_conv1_channel = "stages_3_skips_0_conv1_channel"
stages_3_skips_0_conv2_kernel = "stages_3_skips_0_conv2_kernel"
stages_3_skips_1_conv1_channel = "stages_3_skips_1_conv1_channel"
stages_3_skips_1_conv2_kernel = "stages_3_skips_1_conv2_kernel"
stages_3_skips_2_conv1_channel = "stages_3_skips_2_conv1_channel"
stages_3_skips_2_conv2_kernel = "stages_3_skips_2_conv2_kernel"
stages_4_block_conv1_channel = "stages_4_block_conv1_channel"
stages_4_block_conv2_kernel = "stages_4_block_conv2_kernel"
stages_4_block_conv3_channel = "stages_4_block_conv3_channel"
stages_4_skips_0_conv1_channel = "stages_4_skips_0_conv1_channel"
stages_4_skips_0_conv2_kernel = "stages_4_skips_0_conv2_kernel"
stages_4_skips_1_conv1_channel = "stages_4_skips_1_conv1_channel"
stages_4_skips_1_conv2_kernel = "stages_4_skips_1_conv2_kernel"
stages_4_skips_2_conv1_channel = "stages_4_skips_2_conv1_channel"
stages_4_skips_2_conv2_kernel = "stages_4_skips_2_conv2_kernel"
conv_out_channel = "conv_out_channel"

if __name__ == "__main__":
    pass
