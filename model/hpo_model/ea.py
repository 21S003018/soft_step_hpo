from sko.GA import GA
from sko.PSO import PSO
from const import *
import json


class EAPolicy():
    def __init__(self, search_space: str, observe, algorithm="GA") -> None:
        self.search_space = search_space
        with open(self.search_space, "r") as f:
            self.config = json.load(f)
        self.opt_model = None
        self.opt_loss = 1e6
        self.tag = algorithm
        self.observe = observe
        self.iter = 0
        if self.tag == "GA":
            self.controller = GA(func=self.schaffer, n_dim=44, size_pop=4, max_iter=50, prob_mut=0.001, lb=[1]*44, ub=[32.99, 64.99, 3.99, 32.99, 192.99, 3.99, 32.99, 192.99, 3.99, 192.99, 3.99, 192.99, 3.99, 64.99, 384.99, 3.99,
                                 384.99, 3.99, 384.99, 3.99, 96.99, 576.99, 3.99, 576.99, 3.99, 576.99, 3.99, 128.99, 768.99, 3.99, 768.99, 3.99, 768.99, 3.99, 768.99, 3.99, 256.99, 1536.99, 3.99, 1536.99, 3.99, 1536.99, 3.99, 1024.99, ], precision=1e-7)
        if self.tag == "PSO":
            self.controller = PSO(func=self.schaffer, n_dim=44, size_pop=4, max_iter=50, prob_mut=0.001, lb=[1]*44, ub=[32.99, 64.99, 3.99, 32.99, 192.99, 3.99, 32.99, 192.99, 3.99, 192.99, 3.99, 192.99, 3.99, 64.99, 384.99, 3.99,
                                  384.99, 3.99, 384.99, 3.99, 96.99, 576.99, 3.99, 576.99, 3.99, 576.99, 3.99, 128.99, 768.99, 3.99, 768.99, 3.99, 768.99, 3.99, 768.99, 3.99, 256.99, 1536.99, 3.99, 1536.99, 3.99, 1536.99, 3.99, 1024.99, ], precision=1e-7)

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
        self.iter += 1
        train_loss = self.observe(config)
        return train_loss
