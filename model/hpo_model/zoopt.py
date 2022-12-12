from zoopt import Dimension, Objective, Parameter, Opt
from const import *
import json
import random


class ZooptPolicy():
    def __init__(self, search_space: str, observe) -> None:
        self.search_space = search_space
        with open(self.search_space, "r") as f:
            self.config = json.load(f)
        self.opt_model = None
        self.opt_loss = 1e6
        self.tag = "bayes"
        self.observe = observe
        self.iter = 0
        self.dim = Dimension(44, [[1, 32], [1, 64], [1, 3], [1, 32], [1, 192], [1, 3], [1, 32], [1, 192], [1, 3], [1, 192], [1, 3], [1, 192], [1, 3], [1, 64], [1, 384], [1, 3], [1, 384], [1, 3], [1, 384], [1, 3], [1, 96], [1, 576], [1, 3], [1, 576], [1, 3], [1, 576], [1, 3], [1, 128], [1, 768], [1, 3], [1, 768], [1, 3], [1, 768], [1, 3], [1, 768], [1, 3], [1, 256], [1, 1536], [
                             1, 3], [1, 1536], [1, 3], [1, 1536], [1, 3], [1, 1024]], [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False])
        self.obj = Objective(self.objective, self.dim)

    def objective(self, solution):
        hparams = solution.get_x()
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
        self.iter += 1
        train_loss = self.observe(config)
        return train_loss
