import numpy as np
from math import log, ceil
from const import *
import json
import random


class HyperbandPolicy:
    def __init__(self, search_space, observe):
        self.search_space = search_space
        with open(self.search_space, "r") as f:
            self.config = json.load(f)
        self.tag = "bandit"
        self.observe = observe
        self.iter = 0

        self.max_iter = 81
        self.eta = 3

        self.logeta = lambda x: log(x) / log(self.eta)
        self.s_max = int(self.logeta(self.max_iter))
        self.B = (self.s_max + 1) * self.max_iter

        self.counter = 0
        self.opt_loss = np.inf
        self.opt_config = None
        self.best_counter = -1

    def run(self):
        for s in reversed(range(self.s_max + 1)):
            n = int(ceil(self.B / self.max_iter / (s + 1) * self.eta ** s))
            r = self.max_iter * self.eta ** (-s)
            candidates = [self.sample() for _ in range(n)]
            for i in range(s):
                n_configs = n * self.eta ** (-i)
                n_iterations = int(r * self.eta ** (i))
                train_losses = []
                for config in candidates:
                    self.counter += 1
                    train_loss = self.observe(config, n_iterations)
                    train_losses.append(train_loss)
                indices = np.argsort(train_losses)
                candidates = [candidates[i] for i in indices]
                candidates = candidates[0:int(n_configs / self.eta)]
            config = candidates[0]
            self.iter += 1
            train_loss = self.observe(config)
            if train_loss < self.opt_loss:
                self.opt_loss = train_loss
                self.opt_config = config
        return self.opt_config

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
                output_channel = stage["c"]
                stage_conf["skip"] = []
                input_channel = output_channel
                for _ in range(stage["n"]-1):
                    stage_conf["skip"].append({
                        "type": "skip",
                        "c1": random.randint(1, stage["e"]*input_channel),
                        "k": random.randint(1, int(stage["k"]/2))*2+1,
                        "c2": stage_conf["block"]["c2"],
                        "s": 1
                    })
                config["stage"].append(stage_conf)
        else:
            print("Only linear space evaluated for random policy!")
        return config
