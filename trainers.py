import json
import warnings
import numpy as np
from sklearn.metrics import precision_recall_fscore_support as metrics
import torch
import torch.nn.functional as F
from torch import autograd
from const import *
from utils import Data, num_image
from time import time
from torchstat import stat
from model.cnn_model.resnet import ResNet
from model.cnn_model.mobilenet import MobileNetV2
from model.cnn_model.eval import Eval
from model.nas_model.soft_step import SoftStep
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
        self.save=False
        pass

    def train(self,load=False,save=False):
        if load:
            self.load_model()
        self.save=save
        self.optimizer = torch.optim.SGD(
            self.model.parameters(), lr=0.1, momentum=P_MOMENTUM, weight_decay=1e-4)
        lr_schedular = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer, milestones=[EPOCHS * 0.5, EPOCHS * 0.75], gamma=0.1)
        opt_accu = -1
        for i in range(EPOCHS):
            self.model.train()
            loss_sum = 0
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
                if self.save:
                    self.save_model()
                print(
                    f"Epoch~{i+1}->train_loss:{round(loss_sum,4)}, val_loss:{round(val_loss, 4)}, val_accu:{round(val_accu, 4)}, time:{round(time()-st_time,4)}")
            else:
                print(f"Epoch~{i+1}->time:{round(time()-st_time,4)}")
            lr_schedular.step()
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

    def save_model(self):
        torch.save(self.model.state_dict(), self.save_model_path)
        return

    def load_model(self):
        state_dict = torch.load(self.save_model_path)
        self.model.load_state_dict(state_dict)
        return

class EvalTrainer(CNNTrainer):
    """
    specify for a dataset and a model
    """

    def __init__(self, dataset, path=None) -> None:
        # data
        self.dataset = dataset
        self.train_loader, self.test_loader, self.input_channel, self.inputdim, self.nclass = Data().get(dataset)
        self.num_image = num_image(self.train_loader)
        # model
        self.model = Eval(self.input_channel,self.inputdim,self.nclass,path)
        if torch.cuda.is_available():
            self.model.cuda(DEVICE)
        path_item = path.split("/")[-1]
        self.save_model_path = f"ckpt/{path_item}_{dataset}"
        pass


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
        lr_schedular = torch.optim.lr_scheduler.MultiStepLR(
            self.model_optimizer, milestones=[EPOCHS * 0.5, EPOCHS * 0.75], gamma=0.1)
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


class SoftStepTrainer(CNNTrainer):
    def __init__(self, model_name, dataset, path=None) -> None:
        # load data
        self.dataset = dataset
        self.train_loader, self.test_loader, self.input_channel, self.inputdim, self.nclass = Data().get(dataset)
        self.num_image = num_image(self.train_loader)
        # init model
        self.path = path
        self.model_name = model_name
        self.model = SoftStep(self.input_channel,
                              self.inputdim, self.nclass, path=path)
        if torch.cuda.is_available():
            self.model.cuda(DEVICE)
        self.save_model_path = f"ckpt/{self.model_name}_{self.dataset}"
        return

    def train(self):
        self.model_optimizer = torch.optim.SGD(
            self.model.model_parameters(), lr=0.1, momentum=P_MOMENTUM, weight_decay=1e-4)
        self.arch_optimizer = torch.optim.SGD(
            self.model.arch_parameters(), lr=0.5, momentum=P_MOMENTUM)
        # self.arch_optimizer = torch.optim.Adam(
        #     self.model.arch_parameters(), lr=0.001)
        arch_lr_schedular = torch.optim.lr_scheduler.MultiStepLR(
            self.arch_optimizer, milestones=[EPOCHS * 0.1], gamma=0.5)
        opt_accu = -1
        for i in range(EPOCHS*100):
            self.model.train()
            loss_sum = 0
            st_time = time()
            for imgs, label in self.train_loader:
                if torch.cuda.is_available():
                    imgs = imgs.cuda(DEVICE)
                    label = label.cuda(DEVICE)
                preds = self.model(imgs)
                loss = F.cross_entropy(preds, label)
                self.model_optimizer.zero_grad()
                self.arch_optimizer.zero_grad()
                # # track internal variable's grad
                # indicator_grad_8_conv1 = autograd.grad(
                #     loss, self.model.blocks[8].hidden_indicators, retain_graph=True)[0]
                # controller_grad_8_conv1 = []
                # for indicator in self.model.blocks[8].hidden_indicators:
                #     controller_grad_8_conv1.append(
                #         autograd.grad(indicator, self.model.blocks[8].conv1.channel_alpha, retain_graph=True)[0])
                # for j, indicator_grad in enumerate(indicator_grad_8_conv1):
                #     print(j, self.model.blocks[8].hidden_indicators[j].item(), indicator_grad.item(),
                #           controller_grad_8_conv1[j].item())
                loss.backward()
                if i % 2 == 1:
                    self.arch_optimizer.step()
                else:
                    self.model_optimizer.step()
                loss_sum += loss.item() * len(imgs)/self.num_image
            # arch_lr_schedular.step()
            # eval
            val_accu, val_loss = self.val()
            if val_accu > opt_accu:
                # self.save_model()
                opt_accu = val_accu
            print(f"Epoch~{i+1}->train_loss:{round(loss_sum,4)}, val_loss:{round(val_loss, 4)}, val_accu:{round(val_accu, 4)}, time:{round(time()-st_time,4)}")
            # show arch params
            for name, param in self.model.named_parameters():
                if name.__contains__("alpha"):
                    print(name, param.item(), param.grad.item())
            # save arch params
            current_struc = self.generate_struc()
            with open(f"log/softstep/{i+1}.json", "w") as f:
                json.dump(current_struc, f)
        return

    def generate_struc(self):
        # prepare a body for struc
        with open(self.path, 'r') as f:
            net_struc = json.load(f)
        blocks = []
        for block in net_struc["blocks"]:
            for i in range(block["n"]):
                blocks.append(
                    {"e": block["e"], "c": block["c"], "n": 1, "k": block["k"], "s": block["s"] if i == 0 else 1})
        alphas = self.model.search_result_list()
        alphas = list(alphas)
        alphas = np.array(alphas).reshape((int(len(alphas)/3), 3))
        c_last = net_struc["b0"]["conv_in"]
        for i, alpha_item in enumerate(alphas):
            c1, k, c2 = alpha_item
            e = c1/c_last
            c = int(c2)
            k = 2*max(int(k), 0) + 1
            c_last = c2
            blocks[i]["e"], blocks[i]["c"], blocks[i]["k"] = e, c, k
        net_struc["blocks"] = blocks
        # with open("config/{}-softstep_opt.json".format(self.dataset), "w") as f:
        #     json.dump(net_struc,f)
        return net_struc


if __name__ == "__main__":
    trainer = EvalTrainer(CIFAR100, path='config/softstep_eval.json')
    print(stat(trainer.model,(3,32,32)))

    # trainer = CNNTrainer(MOBILENET,CIFAR100)
    # print(stat(trainer.model,(3,32,32)))

    # trainer = SoftStepTrainer(SOFTSTEP, CIFAR10, path=SEARCHSPACE)
    # trainer.generate_struc()
    pass
