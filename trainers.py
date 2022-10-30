import json
import warnings
import numpy as np
from sklearn.metrics import precision_recall_fscore_support as metrics
import torch
import torch.nn.functional as F
from const import *
from utils import Data, num_image
from time import time
from model.cnn_model.resnet import ResNet
from model.cnn_model.mobilenet import MobileNetV2
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
        pass

    def train(self):
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
            val_accu, _, _, _, val_loss = self.val()
            if val_accu > opt_accu:
                opt_accu = val_accu
                # self.save_model()
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
        preds = []
        Y = []
        for imgs, label in self.test_loader:
            if torch.cuda.is_available():
                imgs = imgs.cuda(DEVICE)
                label = label.cuda(DEVICE)
            tmp_pred = self.model(imgs)
            tmp = tmp_pred.detach().cpu().numpy()
            preds.extend([np.argmax(tmp[i]) for i in range(len(tmp))])
            Y.extend(label.detach().cpu().numpy())
            ncorrect += torch.sum(tmp_pred.max(1)[1].eq(label).double())
            nsample += len(label)
            loss = F.cross_entropy(tmp_pred, label)
            valloss += loss.item() * len(imgs)
        p, r, f1, _ = metrics(preds, Y)
        valloss = valloss/nsample
        return float((ncorrect/nsample).cpu()), p, r, f1, valloss

    def get_metrics(self):
        self.load_model()
        self.model.eval()
        ncorrect = 0
        nsample = 0
        for imgs, label in self.test_loader:
            if torch.cuda.is_available():
                imgs = imgs.cuda(DEVICE)
                label = label.cuda(DEVICE)
            preds = self.model(imgs)
            ncorrect += torch.sum(preds.max(1)[1].eq(label).double())
            nsample += len(label)
        return float((ncorrect/nsample).cpu())

    def save_model(self):
        torch.save(self.model.state_dict(), self.save_model_path)
        return

    def load_model(self):
        state_dict = torch.load(self.save_model_path)
        self.model.load_state_dict(state_dict)
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
            val_accu, _, _, _, val_loss = self.val()
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
            val_accu, _, _, _, val_loss = self.val()
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

    def generate_struc(self):
        # prepare a body for struc
        with open(self.path, 'r') as f:
            net_struc = json.load(f)
        blocks = []
        for block in net_struc["blocks"]:
            for _ in range(block["n"]):
                blocks.append(
                    {"e": block["e"], "c": block["c"], "n": 1, "k": block["k"], "s": block["s"]})
        alphas = []
        for name, param in self.model.named_parameters():
            if name.__contains__('weight') or name.__contains__('bias'):
                continue
            alphas.append(param.item())
        # print(alphas)
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
    trainer = SoftStepTrainer(SOFTSTEP, CIFAR100, path=SEARCHSPACE)
    trainer.generate_struc()
    pass
