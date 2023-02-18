import json
import warnings
import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from const import *
from utils import Data, num_image
from time import time
from torchstat import stat
# import thop,torchsummary
from model.cnn_model.resnet import ResNet
from model.cnn_model.mobilenet import MobileNetV2
from model.cnn_model.eval import Eval, BottleneckEval, ShallowEval, get_block_type
from model.hpo_model.supernet import LinearSupernet, BottleneckSupernet, ShallowSupernet
from model.hpo_model.rand import RandPolicy
from model.hpo_model.bayes import BayesPolicy
from model.hpo_model.zoopt import ZooptPolicy, Parameter, Opt
from model.hpo_model.bandit import HyperbandPolicy
from model.hpo_model.ea import EAPolicy
from model.nas_model.softstep import SoftStep, BottleneckSoftStep, ShallowSoftStep
from model.nas_model.darts import DARTS
from model.nas_model.chamnet import ChamNet
from model.nas_model.mnasnet import PolicyNetwork, ValueNetwork, Environment
warnings.filterwarnings("ignore")


class CNNTrainer():
    """
    specify for a dataset and a model
    """

    def __init__(self, model_name, dataset, device="cuda:0") -> None:
        self.device = device
        # data
        self.dataset = dataset
        self.train_loader, self.test_loader, self.input_channel, self.inputdim, self.nclass = Data().get(dataset)
        self.num_image = num_image(self.train_loader)
        # model
        self.model_name = model_name
        self.model = eval(self.model_name)(
            self.input_channel, self.inputdim, self.nclass)
        if torch.cuda.is_available():
            self.model.cuda(self.device)
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
                    imgs = imgs.cuda(self.device)
                    label = label.cuda(self.device)
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
                imgs = imgs.cuda(self.device)
                label = label.cuda(self.device)
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

    def __init__(self, dataset, path: str = None, device="cuda:0") -> None:
        self.device = device
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
            self.model.cuda(self.device)
        path_item = path.split("/")[-1]
        path_item = path_item.replace(".json", "")
        self.save_model_path = f"ckpt/{path_item}_{dataset}.pkl"
        pass


class SoftStepTrainer(CNNTrainer):
    def __init__(self, model_name, dataset, path=None, opt_order=1, device="cuda:0") -> None:
        # config
        self.path = path
        self.model_name = model_name
        self.dataset = dataset
        self.order = opt_order
        self.device = device
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
            self.model.cuda(self.device)
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
                    imgs = imgs.cuda(self.device)
                    label = label.cuda(self.device)
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


class HPOTrainer(CNNTrainer):
    def __init__(self, policy_name, dataset, search_space=None, device="cuda:0") -> None:
        # config
        self.path = search_space
        self.policy_name = policy_name
        self.dataset = dataset
        self.device = device
        # load data
        self.train_loader, self.test_loader, self.input_channel, self.inputdim, self.nclass = Data().get(dataset)
        self.num_image = num_image(self.train_loader)
        # init model
        if search_space == LINEARSEARCHSPACE:
            self.model = LinearSupernet(self.input_channel,
                                        self.inputdim, self.nclass)
        elif search_space == BOTTLENECKSEARCHSPACE:
            self.model = BottleneckSupernet(self.input_channel,
                                            self.inputdim, self.nclass)
        elif search_space == SHALLOWSEARCHSPACE:
            self.model = ShallowSupernet(self.input_channel,
                                         self.inputdim, self.nclass)
        if torch.cuda.is_available():
            self.model.cuda(self.device)
        # pretrained model path
        self.pretrained_model_path = "ckpt/pretrained_supernet_{}".format(
            'cifar10' if self.dataset == CIFAR10 else 'cifar100')
        # init policy
        if self.policy_name == "rand":
            self.policy = RandPolicy(search_space)
        if self.policy_name == "bayes":
            self.policy = BayesPolicy(search_space, self.observe)
        if self.policy_name == "zoopt":
            self.policy = ZooptPolicy(search_space, self.observe)
        if self.policy_name == "bandit":
            self.policy = HyperbandPolicy(search_space, self.observe)
        if self.policy_name == "ga":
            self.policy = EAPolicy(search_space, self.observe, "ga")
        if self.policy_name == "pso":
            self.policy = EAPolicy(search_space, self.observe, "ga")
        return

    def rand_search(self):
        # totally search for 200 iters and 20 rounds for observation's train
        for t in range(200):
            # st_time = time()
            self.policy.iter += 1
            config = self.policy.sample()
            train_loss = self.observe(config)
            self.policy.step(train_loss, self.model.generate_config())
            # ed_time = time()
            # val_accu, val_loss = self.val()
            # print(f"Episode~{t+1}->train_loss:{round(train_loss,4)},val_loss:{round(val_loss, 4)}, val_accu:{round(val_accu, 4)}, time:{round(ed_time-st_time,4)}")
            # with open("log/rand/{}_{}.json".format(t+1, self.dataset), "w") as f:
            #     json.dump(self.model.generate_config(), f)
        # save the opt model
        # with open("search_result/{}_linear_cifar10.json".format(self.policy.tag), "w") as f:
        #     json.dump(self.policy.opt_model, f)
        return

    def bayes_search(self):
        st_time = time()
        self.policy.controller.maximize(n_iter=200)
        ed_time = time()
        print(ed_time-st_time, self.policy.controller.max["params"],
              self.policy.controller.max["target"], self.policy.controller.res)
        return

    def zoopt_search(self):
        st_time = time()
        solution = Opt.min(self.policy.obj, Parameter(budget=200,))
        ed_time = time()
        print(ed_time-st_time, solution.get_x())
        return

    def bandit_search(self):
        st_time = time()
        opt_config = self.policy.run()
        ed_time = time()
        print(ed_time-st_time, opt_config)
        return

    def ea_search(self):
        st_time = time()
        opt_config, opt_loss = self.policy.controller.run()
        ed_time = time()
        print(ed_time-st_time, opt_config, opt_loss)
        return

    def observe(self, config, niter=20):
        st_time = time()
        self.load_model(self.pretrained_model_path)
        self.model.update_indicators(config)
        optimizer = torch.optim.SGD(self.model.parameters(
        ), lr=0.1, momentum=P_MOMENTUM, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, niter, eta_min=0.001)
        for _ in range(niter):
            self.model.train()
            scheduler.step()
            train_loss = 0
            for imgs, label in self.train_loader:
                if torch.cuda.is_available():
                    imgs = imgs.cuda(self.device)
                    label = label.cuda(self.device)
                preds = self.model(imgs)
                loss = F.cross_entropy(preds, label)
                optimizer.zero_grad()
                loss.backward()
                train_loss += loss.item()*len(imgs)/self.num_image
                optimizer.step()
        ed_time = time()
        val_accu, val_loss = self.val()
        print(f"Episode~{self.policy.iter}->train_loss:{round(train_loss,4)},val_loss:{round(val_loss, 4)}, val_accu:{round(val_accu, 4)}, time:{round(ed_time-st_time,4)}")

        # if self.policy_name in ["bayes", "zoopt"]:
        with open("log/{}/{}_{}.json".format(self.policy.tag, self.policy.iter, self.dataset), "w") as f:
            json.dump(self.model.generate_config(), f)
        return train_loss

    def pre_train(self):
        # self.load_model(self.pretrained_model_path)
        self.model.update_indicators(self.policy.sample(), True)
        optimizer = torch.optim.SGD(
            self.model.parameters(), lr=0.1, momentum=P_MOMENTUM, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, EPOCHS, eta_min=0.001)
        for i in range(EPOCHS):
            self.model.train()
            loss_sum = 0
            scheduler.step()
            st_time = time()
            for imgs, label in self.train_loader:
                if torch.cuda.is_available():
                    imgs = imgs.cuda(self.device)
                    label = label.cuda(self.device)
                preds = self.model(imgs)
                loss = F.cross_entropy(preds, label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_sum += loss.item() * len(imgs)/self.num_image
            ed_time = time()
            epoch_note = ''
            print(
                f"({epoch_note})Epoch~{i+1}->train_loss:{round(loss_sum,4)}, time:{round(ed_time-st_time,4)}, learning rate:{round(scheduler.get_lr()[0],4)}")
        # save model
        self.save_model(self.pretrained_model_path)
        return


class NasTrainer(CNNTrainer):
    def __init__(self, model_name, dataset, search_space=None, device="cuda:0") -> None:
        self.device = device
        # load data
        self.dataset = dataset
        self.train_loader, self.test_loader, self.input_channel, self.inputdim, self.nclass = Data().get(dataset)
        self.num_image = num_image(self.train_loader)
        # init model
        self.model_name = model_name
        if model_name == "darts":
            self.model = DARTS(self.input_channel,
                               self.inputdim, self.nclass, path=search_space)
        if model_name == "chamnet" and search_space == LINEARSEARCHSPACE:
            self.model = LinearSupernet(self.input_channel,
                                        self.inputdim, self.nclass)
            self.policy = ChamNet(search_space, self.observe)
        if torch.cuda.is_available():
            self.model.cuda(self.device)
        self.save_model_path = f"ckpt/{self.model_name}_{self.dataset}"
        self.flag = 0
        return

    def darts_search(self):
        self.model_optimizer = torch.optim.SGD(
            self.model.model_parameters(), lr=0.1, momentum=P_MOMENTUM, weight_decay=1e-4)
        self.arch_optimizer = torch.optim.SGD(
            self.model.arch_parameters(), lr=0.1, momentum=P_MOMENTUM, weight_decay=1e-4)
        schedular = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.model_optimizer, EPOCHS, eta_min=0.001)
        opt_accu = -1
        for i in range(EPOCHS):
            self.model.train()
            loss_sum = 0
            st_time = time()
            schedular.step()
            for imgs, label in self.train_loader:
                if torch.cuda.is_available():
                    imgs = imgs.cuda(self.device)
                    label = label.cuda(self.device)
                preds = self.model(imgs)
                loss = F.cross_entropy(preds, label)
                self.model_optimizer.zero_grad()
                loss.backward()
                self.model_optimizer.step()
                preds = self.model(imgs)
                loss = F.cross_entropy(preds, label)
                self.arch_optimizer.zero_grad()
                loss.backward()
                self.arch_optimizer.step()
                loss_sum += loss.item() * len(imgs)/self.num_image
            ed_time = time()
            # eval
            val_accu, val_loss = self.val()
            if val_accu > opt_accu:
                self.save_model()
                opt_accu = val_accu
                print(
                    f"Epoch~{i+1}->train_loss:{round(loss_sum,4)}, val_loss:{round(val_loss, 4)}, val_accu:{round(val_accu, 4)}, time:{round(ed_time-st_time,4)}")
            # save model
            with open(f"log/{self.model_name}/{i+1}_{self.dataset}.json", "w") as f:
                json.dump(self.model.generate_config(), f)
        return

    def chamnet_search(self):
        # train the predictor
        # self.policy.train_predictor()
        # ea search
        self.policy.train_predictor(True)
        st_time = time()
        opt_config, opt_loss = self.policy.controller.run()
        ed_time = time()
        print(ed_time-st_time, opt_config, opt_loss)
        return

    def observe(self, config, niter=200):
        st_time = time()
        self.model.update_indicators(config)
        optimizer = torch.optim.SGD(self.model.parameters(
        ), lr=0.1, momentum=P_MOMENTUM, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, niter, eta_min=0.001)
        for _ in range(niter):
            self.model.train()
            scheduler.step()
            train_loss = 0
            for imgs, label in self.train_loader:
                if torch.cuda.is_available():
                    imgs = imgs.cuda(self.device)
                    label = label.cuda(self.device)
                preds = self.model(imgs)
                loss = F.cross_entropy(preds, label)
                optimizer.zero_grad()
                loss.backward()
                train_loss += loss.item()*len(imgs)/self.num_image
                optimizer.step()
        ed_time = time()
        val_accu, val_loss = self.val()
        print(f"Episode~{self.policy.iter}->train_loss:{round(train_loss,4)},val_loss:{round(val_loss, 4)}, val_accu:{round(val_accu, 4)}, time:{round(ed_time-st_time,4)}")
        return train_loss


class RLTrainer(NasTrainer):
    def __init__(self, model_name, dataset, search_space=None, device="cuda:0") -> None:
        self.device = device
        # load data
        self.dataset = dataset
        self.train_loader, self.test_loader, self.input_channel, self.inputdim, self.nclass = Data().get(dataset)
        self.num_image = num_image(self.train_loader)
        # init model
        self.model_name = model_name
        if self.model_name == "mnasnet":
            self.policy_model = PolicyNetwork()
            self.value_model = ValueNetwork()
            self.environment = Environment(search_space)
        if torch.cuda.is_available():
            self.policy_model.cuda(self.device)
            self.value_model.cuda(self.device)
        # self.save_model_path = f"ckpt/{self.model_name}_{self.dataset}"
        self.flag = 0
        return

    def mnasnet_search(self):
        length_episodes = 512
        num_processes = 8
        curr_state = torch.Tensor(
            [self.environment.sample_state() for _ in range(num_processes)])
        if torch.cuda.is_available():
            curr_state = curr_state.cuda(self.device)
        while True:
            states = []
            values = []
            rewards = []
            actions = []
            for _ in range(length_episodes):
                states.extend(curr_state.clone())
                value = self.value_model(curr_state)
                values.extend(value)
                logits = self.policy_model(curr_state)
                action = []
                for logit in logits:
                    policy = F.softmax(logit, dim=1)
                    action.append(Categorical(policy).sample())
                action = torch.stack(action).T
                actions.extend(action)
                next_states = []
                for i, state in enumerate(curr_state):
                    next_state = self.environment.next_state(
                        state, action[i]-1)
                    next_state_config = self.environment.get_config(next_state)
                    train_loss = self.observe(next_state_config)
                    reward = 1 - train_loss
                    rewards.append(reward)
                    next_states.append(next_state)
                curr_state = torch.Tensor(next_states)
                if torch.cuda.is_available():
                    curr_state = curr_state.cuda(self.device)

                break
            break
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
    # model = ShallowEval(
    #     3, 32, 100, path='log/softstep_shallow_cifar100/196_o1_cifar-100-python.json')
    # model = BottleneckEval(
    #     3, 32, 100, path='search_result/softstep_bottleneck_cifar100.json')
    # model = BottleneckEval(
    #     3, 32, 100, path='log/softstep_bottleneck_cifar100/192_o1_cifar-100-python.json')
    # model = BottleneckEval(
    #     3, 32, 100, path='config/search_space_bottleneck_eval.json')
    # model = Eval(3, 32, 100, path='config/search_space_linear_eval.json')
    # model = Eval(
    #     3, 32, 100, path='search_result/softstep_linear_cifar100.json')
    # model = ResNet(3, 32, 100)
    # model = MobileNetV2(3, 32, 100)
    # print(stat(model, (3, 32, 32)))
    # thop.profile(model, inputs=torch.randn((1,3,32,32)))
    # torchsummary.summary(model,(3,32,32))

    # policy = RandPolicy(LINEARSEARCHSPACE)
    # with open("test.json", "w") as f:
    #     json.dump(policy.sample(), f)

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

    # model = PolicyNetwork()
    # x = torch.rand((32, 44))
    # distris = model(x)
    # print(distris)
    # for distri in distris:
    #     print(distri.size())
    #     print(distri)
    # print(x.size())
    # print(x)
    x = torch.zeros((3, 4))
    print(x)
    print(F.softmax(x, dim=0))
    pass
