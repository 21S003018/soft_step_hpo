import json
import warnings
import numpy as np
import torch
import torch.nn.functional as F
from torch import autograd
from const import *
from utils import Data, num_image
from time import time
from torchstat import stat
# import thop,torchsummary
from model.cnn_model.resnet import ResNet
from model.cnn_model.mobilenet import MobileNetV2
from model.cnn_model.eval import Eval, BottleneckEval, ShallowEval, get_block_type
from model.nas_model.softstep import SoftStep, BottleneckSoftStep, ShallowSoftStep
warnings.filterwarnings("ignore")


class CNNTrainer():
    """
    specify for a dataset and a model
    """

    def __init__(self, model_name, dataset) -> None:
        # data
        self.dataset = dataset
        self.train_loader, self.test_loader, self.input_channel, self.inputdim, self.nclass = Data().get(dataset)
        self.num_image = num_image(self.train_loader)
        # model
        self.model_name = model_name
        self.model = eval(self.model_name)(
            self.input_channel, self.inputdim, self.nclass)
        if torch.cuda.is_available():
            self.model.cuda(DEVICE)
        self.save_model_path = f"ckpt/{self.model_name}_{self.dataset}"
        pass

    def train(self, load=False, save=False, epochs=EPOCHS):
        if load:
            self.load_model()
        self.optimizer = torch.optim.SGD(
            self.model.parameters(), lr=0.1, momentum=P_MOMENTUM, weight_decay=1e-4)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, EPOCHS, eta_min=0.001)
        opt_accu = -1
        for i in range(epochs):
            self.model.train()
            loss_sum = 0
            lr_scheduler.step()
            st_time = time()
            for imgs, label in self.train_loader:
                if torch.cuda.is_available():
                    imgs = imgs.cuda(DEVICE)
                    label = label.cuda(DEVICE)
                preds = self.model(imgs)
                loss = F.cross_entropy(preds, label)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                loss_sum += loss.item() * len(imgs)/self.num_image
            # eval
            val_accu, val_loss = self.val()
            if val_accu > opt_accu:
                opt_accu = val_accu
                # if save and i < epochs/2:  # special save
                #     # if save:
                #     self.save_model()
                print(
                    f"Epoch~{i+1}->train_loss:{round(loss_sum,4)}, val_loss:{round(val_loss, 4)}, val_accu:{round(val_accu, 4)}, time:{round(time()-st_time,4)}")
            else:
                print(f"Epoch~{i+1}->time:{round(time()-st_time,4)}")
        return

    def val(self):
        self.model.eval()
        ncorrect = 0
        nsample = 0
        valloss = 0
        for imgs, label in self.test_loader:
            if torch.cuda.is_available():
                imgs = imgs.cuda(DEVICE)
                label = label.cuda(DEVICE)
            preds = self.model(imgs)
            ncorrect += torch.sum(preds.max(1)[1].eq(label).double())
            nsample += len(label)
            loss = F.cross_entropy(preds, label)
            valloss += loss.item() * len(imgs)
        valloss = valloss/nsample
        return float((ncorrect/nsample).cpu()), valloss

    def save_model(self, path=None):
        if path:
            torch.save(self.model.state_dict(), path)
        else:
            torch.save(self.model.state_dict(), self.save_model_path)
        return

    def load_model(self, path=None):
        if path:
            state_dict = torch.load(path)
        else:
            state_dict = torch.load(self.save_model_path)
        self.model.load_state_dict(state_dict)
        return


class EvalTrainer(CNNTrainer):
    """
    specify for a dataset and a model
    """

    def __init__(self, dataset, path: str = None) -> None:
        # data
        self.dataset = dataset
        self.train_loader, self.test_loader, self.input_channel, self.inputdim, self.nclass = Data().get(dataset)
        self.num_image = num_image(self.train_loader)
        # model
        block_type = get_block_type(path)
        if block_type == "linear":
            self.model = Eval(self.input_channel,
                              self.inputdim, self.nclass, path)
        elif block_type == "bottleneck":
            self.model = BottleneckEval(
                self.input_channel, self.inputdim, self.nclass, path)
        elif block_type == "shallow":
            self.model = ShallowEval(
                self.input_channel, self.inputdim, self.nclass, path)

        if torch.cuda.is_available():
            self.model.cuda(DEVICE)
        path_item = path.split("/")[-1]
        path_item = path_item.replace(".json", "")
        self.save_model_path = f"ckpt/{path_item}_{dataset}.pkl"
        pass


class SoftStepTrainer(CNNTrainer):
    def __init__(self, model_name, dataset, path=None, opt_order=1) -> None:
        # config
        self.path = path
        self.model_name = model_name
        self.dataset = dataset
        self.order = opt_order
        self.arch_decay = 1e-5 if self.dataset == CIFAR10 else 1e-5
        self.arch_lr = 0.1
        # load data
        self.train_loader, self.test_loader, self.input_channel, self.inputdim, self.nclass = Data().get(dataset)
        self.num_image = num_image(self.train_loader)
        # init model
        if path == LINEARSEARCHSPACE:
            self.model = SoftStep(self.input_channel,
                                  self.inputdim, self.nclass, path=path)
        elif path == BOTTLENECKSEARCHSPACE:
            self.model = BottleneckSoftStep(self.input_channel,
                                            self.inputdim, self.nclass, path=path)
        elif path == SHALLOWSEARCHSPACE:
            self.model = ShallowSoftStep(self.input_channel,
                                         self.inputdim, self.nclass, path=path)

        if torch.cuda.is_available():
            self.model.cuda(DEVICE)
        self.model.update_indicators()
        return

    def train(self):
        self.model_optimizer = torch.optim.SGD(
            self.model.model_parameters(), lr=0.1, momentum=P_MOMENTUM, weight_decay=1e-4)
        self.arch_optimizer = torch.optim.SGD(
            self.model.arch_parameters(), lr=self.arch_lr, momentum=P_MOMENTUM, weight_decay=self.arch_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.model_optimizer, EPOCHS, eta_min=0.001)
        opt_accu = -1
        for i in range(EPOCHS):
            self.model.train()
            loss_sum = 0
            scheduler.step()
            st_time = time()
            for imgs, label in self.train_loader:
                if torch.cuda.is_available():
                    imgs = imgs.cuda(DEVICE)
                    label = label.cuda(DEVICE)
                # fine tune model
                for _ in range(self.order):
                    arch_opt = False
                    preds = self.model(imgs, arch_opt)
                    loss = F.cross_entropy(preds, label)
                    self.model_optimizer.zero_grad()
                    loss.backward()
                    self.model_optimizer.step()
                # fine tune arch
                arch_opt = True
                preds = self.model(imgs, arch_opt)
                loss = F.cross_entropy(preds, label)
                self.arch_optimizer.zero_grad()
                loss.backward()
                self.arch_optimizer.step()
                self.model.protect_controller()
                loss_sum += loss.item() * len(imgs)/self.num_image
            ed_time = time()
            # eval
            val_accu, val_loss = self.val()
            if val_accu > opt_accu:
                opt_accu = val_accu
            # show epoch training message
            epoch_note = ''
            # epoch_note = 'Arch' if arch_opt else 'Modl'
            print(
                f"({epoch_note})Epoch~{i+1}->train_loss:{round(loss_sum,4)}, val_loss:{round(val_loss, 4)}, val_accu:{round(val_accu, 4)}, time:{round(ed_time-st_time,4)}, learning rate:{round(scheduler.get_lr()[0],4)}")
            # show arch params
            # if arch_opt:
            for name, param in self.model.named_parameters():
                if name.__contains__("alpha"):
                    print(name, param.item(), param.grad.item())
            # save arch params
            current_config = self.model.generate_config()
            with open(f"log/softstep/{i+1}_o{self.order}_{self.dataset}.json", "w") as f:
                json.dump(current_config, f)
        return


class NasTrainer(CNNTrainer):
    def __init__(self, model_name, dataset, path=None) -> None:
        # load data
        self.dataset = dataset
        self.train_loader, self.test_loader, self.input_channel, self.inputdim, self.nclass = Data().get(dataset)
        self.num_image = num_image(self.train_loader)
        # init model
        self.model_name = model_name
        self.model = SoftStep(self.input_channel,
                              self.inputdim, self.nclass, path=path)
        if torch.cuda.is_available():
            self.model.cuda(DEVICE)
        self.save_model_path = f"ckpt/{self.model_name}_{self.dataset}"
        self.flag = 0
        return

    def train(self):
        self.model_optimizer = torch.optim.SGD(
            self.model.model_parameters(), lr=0.1, momentum=P_MOMENTUM)
        self.arch_optimizer = torch.optim.SGD(
            self.model.arch_parameters(), lr=0.1, momentum=P_MOMENTUM)
        lr_schedular = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.model_optimizer, EPOCHS, eta_min=0.001)
        opt_accu = -1
        for i in range(EPOCHS):
            self.model.train()
            loss_sum = 0
            st_time = time()
            if self.flag % 2 == 0:
                # tune model params
                for imgs, label in self.train_loader:
                    if torch.cuda.is_available():
                        imgs = imgs.cuda(DEVICE)
                        label = label.cuda(DEVICE)
                    preds = self.model(imgs)
                    loss = F.cross_entropy(preds, label)
                    self.model_optimizer.zero_grad()
                    loss.backward()
                    self.model_optimizer.step()
                    loss_sum += loss.item() * len(imgs)/self.num_image
            else:
                # tune arch params
                for imgs, label in self.train_loader:
                    if torch.cuda.is_available():
                        imgs = imgs.cuda(DEVICE)
                        label = label.cuda(DEVICE)
                    preds = self.model(imgs)
                    loss = F.cross_entropy(preds, label)
                    self.arch_optimizer.zero_grad()
                    loss.backward()
                    self.arch_optimizer.step()
                    loss_sum += loss.item() * len(imgs)/self.num_image
            # eval
            val_accu, val_loss = self.val()
            if val_accu > opt_accu:
                self.save_model()
                opt_accu = val_accu
                print(
                    f"Epoch~{i+1}->train_loss:{round(loss_sum,4)}, val_loss:{round(val_loss, 4)}, val_accu:{round(val_accu, 4)}, time:{round(time()-st_time,4)}")
                # # check arch params
                # size_vectors = self.model.generate_size_vector()
                # for tmp in size_vectors:
                #     print(tmp)
                # print()
            lr_schedular.step()
        return


if __name__ == "__main__":
    # trainer = EvalTrainer(CIFAR100, path='search_result/softstep_linear_o1_cifar10.json')
    # trainer = EvalTrainer(CIFAR100, path='test.json')
    # trainer = EvalTrainer(CIFAR100, path='config/search_space_linear_eval.json')
    # print(stat(trainer.model, (3, 32, 32)))
    # model = BottleneckEval(
    #     3, 32, 100, path='config/search_space_bottleneck_eval.json')
    # model = BottleneckSoftStep(3,32,100,BOTTLENECKSEARCHSPACE)
    # model = ShallowSoftStep(3, 32, 100, SHALLOWSEARCHSPACE)
    # config = model.generate_config(full=True)
    # with open("test.json", "w") as f:
    #     json.dump(config, f)

    # model = ShallowEval(
    #     3, 32, 100, path='config/search_space_shallow_eval.json')
    model = BottleneckEval(
        3, 32, 100, path='log/softstep_bottleneck_cifar100/192_o1_cifar-100-python.json')
    # model = BottleneckEval(
    #     3, 32, 100, path='config/search_space_bottleneck_eval.json')
    # model = Eval(3, 32, 100, path='config/search_space_linear_eval.json')
    # model = Eval(
    #     3, 32, 100, path='search_result/softstep_linear_cifar100_1e-5.json')
    # model = ResNet(3, 32, 100)
    # model = MobileNetV2(3, 32, 100)
    print(stat(model, (3, 32, 32)))
    # thop.profile(model, inputs=torch.randn((1,3,32,32)))
    # torchsummary.summary(model,(3,32,32))

    # model = ShallowSoftStep(
    #     3, 32, 100, path='config/search_space_shallow.json')
    # model = BottleneckSoftStep(
    #     3, 32, 100, path='config/search_space_bottleneck.json')
    # with open("test.json", "w") as f:
    #     json.dump(model.generate_config(True), f)

    # trainer = CNNTrainer(MOBILENET,CIFAR100)
    # print(stat(trainer.model,(3,32,32)))

    # trainer = SoftStepTrainer(SOFTSTEP, CIFAR100, path=LINEARSEARCHSPACE)
    # for name, param in trainer.model.named_parameters():
    #     print(name)
    pass
