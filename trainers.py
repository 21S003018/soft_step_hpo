import torch
from torch.optim.adam import Adam
from torch.optim.adamax import Adamax
from const import *
import torch.nn.functional as F
import warnings
from sklearn.metrics import precision_recall_fscore_support as metrics
import numpy as np

from utils import Data, num_image
from time import time
from model.cnn_model.resnet import ResNet
from model.cnn_model.mobilenet import mobilenetv2
from model.cnn_model.shufflenet import ShuffleNet
from model.nas_model.soft_step import SoftStep
warnings.filterwarnings("ignore")


class CNNTrainer():
    """
    specify for a dataset and a model
    """

    def __init__(self, model_name, dataset) -> None:
        # init data
        self.dataset = dataset
        self.train_loader, self.test_loader, input_channel, inputdim, nclass = Data().get(dataset)
        self.num_image = num_image(self.train_loader)  # get num of image
        # init model
        self.model_name = model_name
        if model_name is RESNET:
            self.model = ResNet(input_channel, inputdim, nclass)
        elif model_name is MOBILENET:
            self.model = mobilenetv2()
        elif model_name is SOFTSTEP:
            self.model = SoftStep(input_channel, inputdim, nclass)
        else :
            self.model = ShuffleNet()
        # self.model = eval(self.model_name)(input_channel, inputdim, nclass)
        if torch.cuda.is_available():
            self.model.cuda()
        self.save_model_path = f"ckpt/{self.model_name}_{self.dataset}"
        pass

    def train(self):
        self.model.reset_parameters()
        self.optimizer = torch.optim.SGD(
            self.model.parameters(), lr=0.1, momentum=P_MOMENTUM)
        lr_schedular = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer, milestones=[EPOCHS * 0.5, EPOCHS * 0.75], gamma=0.1)
        opt_accu = -1
        for i in range(EPOCHS):
            self.model.train()
            loss_sum = 0
            st_time = time()
            for imgs, label in self.train_loader:
                if torch.cuda.is_available():
                    imgs = imgs.cuda()
                    label = label.cuda()
                preds = self.model(imgs)
                loss = F.cross_entropy(preds, label)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                loss_sum += loss.item() * len(imgs)/self.num_image
            # eval
            val_accu, _, _, _, val_loss = self.val()
            if val_accu > opt_accu:
                self.save_model()
            print(
                f"Epoch~{i+1}->train_loss:{round(loss_sum,4)}, val_loss:{round(val_loss, 4)}, val_accu:{round(val_accu, 4)}, time:{round(time()-st_time,4)}")
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
                imgs = imgs.cuda()
                label = label.cuda()
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
                imgs = imgs.cuda()
                label = label.cuda()
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
