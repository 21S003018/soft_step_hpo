import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torch.nn.init as init
from const import *
from sko.GA import GA
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import math
import json
import pickle as pkl
import random
import numpy as np


class ChamNet():
    def __init__(self, search_space: str, observe) -> None:
        self.search_space = search_space
        with open(self.search_space, "r") as f:
            self.config = json.load(f)
        self.opt_model = None
        self.opt_loss = 1e6
        self.observe = observe
        self.controller = GA(func=self.schaffer, n_dim=44, size_pop=96, max_iter=100, prob_mut=0.001, lb=[1]*44, ub=[32.99, 64.99, 3.99, 32.99, 192.99, 3.99, 32.99, 192.99, 3.99, 192.99, 3.99, 192.99, 3.99, 64.99, 384.99, 3.99,
                                                                                                                     384.99, 3.99, 384.99, 3.99, 96.99, 576.99, 3.99, 576.99, 3.99, 576.99, 3.99, 128.99, 768.99, 3.99, 768.99, 3.99, 768.99, 3.99, 768.99, 3.99, 256.99, 1536.99, 3.99, 1536.99, 3.99, 1536.99, 3.99, 1024.99, ], precision=1e-7)
        self.predictor = GaussianProcessRegressor(kernel=C(
            constant_value=1) * RBF(length_scale=1, length_scale_bounds=(1e-2, 1e2)),)
        self.predictor_path = "ckpt/chamnet_predictor"
        return

    def schaffer(self, hparams):
        config = {"type": self.config["type"]}
        config = {
            "type": "linear",
            "layer": {
                "conv_in": int(hparams[0]),
                "conv_out": int(hparams[-1])
            },
            "block": {
                "type": "normal",
                "c1": int(hparams[1]),
                "k": int(hparams[2])*2+1,
                "c2": int(hparams[3]),
                "s": 1
            },
            "stage": [
                {
                    "block": {
                        "type": "reduction",
                        "c1": int(hparams[4]),
                        "k": int(hparams[5])*2+1,
                        "c2": int(hparams[6]),
                        "s": 2
                    },
                    "skip": [
                        {
                            "type": "skip",
                            "c1": int(hparams[7]),
                            "k": int(hparams[8])*2+1,
                            "c2": int(hparams[6]),
                            "s": 1
                        },
                        {
                            "type": "skip",
                            "c1": int(hparams[9]),
                            "k": int(hparams[10])*2+1,
                            "c2": int(hparams[6]),
                            "s": 1
                        }
                    ]
                },
                {
                    "block": {
                        "type": "reduction",
                        "c1": int(hparams[11]),
                        "k": int(hparams[12])*2+1,
                        "c2": int(hparams[13]),
                        "s": 2
                    },
                    "skip": [
                        {
                            "type": "skip",
                            "c1": int(hparams[14]),
                            "k": int(hparams[15])*2+1,
                            "c2": int(hparams[13]),
                            "s": 1
                        },
                        {
                            "type": "skip",
                            "c1": int(hparams[16]),
                            "k": int(hparams[17])*2+1,
                            "c2": int(hparams[13]),
                            "s": 1
                        }
                    ]
                },
                {
                    "block": {
                        "type": "reduction",
                        "c1": int(hparams[18]),
                        "k": int(hparams[19])*2+1,
                        "c2": int(hparams[20]),
                        "s": 2
                    },
                    "skip": [
                        {
                            "type": "skip",
                            "c1": int(hparams[21]),
                            "k": int(hparams[22])*2+1,
                            "c2": int(hparams[20]),
                            "s": 1
                        },
                        {
                            "type": "skip",
                            "c1": int(hparams[23]),
                            "k": int(hparams[24])*2+1,
                            "c2": int(hparams[20]),
                            "s": 1
                        }
                    ]
                },
                {
                    "block": {
                        "type": "normal",
                        "c1": int(hparams[25]),
                        "k": int(hparams[26])*2+1,
                        "c2": int(hparams[27]),
                        "s": 1
                    },
                    "skip": [
                        {
                            "type": "skip",
                            "c1": int(hparams[28]),
                            "k": int(hparams[29])*2+1,
                            "c2": int(hparams[27]),
                            "s": 1
                        },
                        {
                            "type": "skip",
                            "c1": int(hparams[30]),
                            "k": int(hparams[31])*2+1,
                            "c2": int(hparams[27]),
                            "s": 1
                        },
                        {
                            "type": "skip",
                            "c1": int(hparams[32]),
                            "k": int(hparams[33])*2+1,
                            "c2": int(hparams[27]),
                            "s": 1
                        }
                    ]
                },
                {
                    "block": {
                        "type": "normal",
                        "c1": int(hparams[34]),
                        "k": int(hparams[35])*2+1,
                        "c2": int(hparams[36]),
                        "s": 1
                    },
                    "skip": [
                        {
                            "type": "skip",
                            "c1": int(hparams[37]),
                            "k": int(hparams[38])*2+1,
                            "c2": int(hparams[36]),
                            "s": 1
                        },
                        {
                            "type": "skip",
                            "c1": int(hparams[39]),
                            "k": int(hparams[40])*2+1,
                            "c2": int(hparams[36]),
                            "s": 1
                        },
                        {
                            "type": "skip",
                            "c1": int(hparams[41]),
                            "k": int(hparams[42])*2+1,
                            "c2": int(hparams[36]),
                            "s": 1
                        }
                    ]
                }
            ]
        }
        # train_loss = self.observe(config)
        predict_loss = self.predictor(hparams)
        return predict_loss

    def sample(self) -> dict:
        if self.search_space == LINEARSEARCHSPACE:
            config = {"type": self.config["type"]}
            # layer
            config["layer"] = {
                "conv_in": random.randint(1, self.config["layer"]["conv_in"]),
                "conv_out": random.randint(1, self.config["layer"]["conv_out"]),
            }
            output_channel = self.config["layer"]["conv_in"]
            # block
            input_channel = output_channel
            config["block"] = {
                "type": self.config["block"]["type"],
                "c1": random.randint(1, self.config["block"]["e"]*input_channel),
                "k": random.randint(1, int(self.config["block"]["k"]/2))*2+1,
                "c2": random.randint(1, self.config["block"]["c"]),
                "s": self.config["block"]["s"]
            }
            output_channel = self.config["block"]["c"]
            x = []
            x.append(config["layer"]["conv_in"])
            x.append(config["block"]["c1"])
            x.append(config["block"]["k"])
            x.append(config["block"]["c2"])
            # stage
            config["stage"] = []
            for i, stage in enumerate(self.config["stage"]):
                stage_conf = {}
                input_channel = output_channel
                stage_conf["block"] = {
                    "type": stage["type"],
                    "c1": random.randint(1, stage["e"]*input_channel),
                    "k": random.randint(1, int(stage["k"]/2))*2+1,
                    "c2": random.randint(1, stage["c"]),
                    "s": stage["s"]
                }
                x.append(stage_conf["block"]["c1"])
                x.append(stage_conf["block"]["k"])
                x.append(stage_conf["block"]["c2"])
                output_channel = stage["c"]
                stage_conf["skip"] = []
                input_channel = output_channel
                for j in range(stage["n"]-1):
                    stage_conf["skip"].append({
                        "type": "skip",
                        "c1": random.randint(1, stage["e"]*input_channel),
                        "k": random.randint(1, int(stage["k"]/2))*2+1,
                        "c2": stage_conf["block"]["c2"],
                        "s": 1
                    })
                    x.append(stage_conf["skip"][j]["c1"])
                    x.append(stage_conf["skip"][j]["k"])
                    x.append(stage_conf["skip"][j]["c2"])
                config["stage"].append(stage_conf)
            x.append(config["layer"]["conv_out"])
        else:
            print("Only linear space evaluated for random policy!")
        return x, config

    def train_predictor(self, load=False):
        if load:
            with open(self.predictor_path, "rb") as f:
                self.predictor = pkl.load(f)
            return
        X = []
        Y = []
        for _ in range(200):
            x, config = self.sample()
            y = self.observe(config)
            X.append(x)
            Y.append(y)
        self.predictor.fit(np.array(X), np.array(Y))
        with open(self.predictor_path, "wb") as f:
            json.dump(self.predictor, f)
        return
