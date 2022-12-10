from const import *
import json
import random

class RandPolicy():
    def __init__(self, search_space: str) -> None:
        self.search_space = search_space
        with open(self.search_space, "r") as f:
            self.config = json.load(f)
        self.opt_model = None
        self.opt_loss = 1e6
        self.tag = "rand"
        pass
    def sample(self)->dict:
        if self.search_space==LINEARSEARCHSPACE:
            config = {"type": self.config["type"]}
            # layer
            config["layer"] = {
                "conv_in": random.randint(1,self.config["layer"]["conv_in"]),
                "conv_out": random.randint(1,self.config["layer"]["conv_out"]),
            }
            output_channel = self.config["layer"]["conv_in"]
            # block
            input_channel = output_channel
            config["block"] = {
                    "type": self.config["block"]["type"],
                    "c1": random.randint(1,self.config["block"]["e"]*input_channel),
                    "k": random.randint(1,int(self.config["block"]["k"]/2))*2+1,
                    "c2": random.randint(1,self.config["block"]["c"]),
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
                    "c1": random.randint(1,stage["e"]*input_channel),
                    "k": random.randint(1,int(stage["k"]/2))*2+1,
                    "c2": random.randint(1,stage["c"]),
                    "s": stage["s"]
                }
                output_channel = stage["c"]
                stage_conf["skip"] = []
                input_channel = output_channel
                for _ in range(stage["n"]-1):
                    stage_conf["skip"].append({
                    "type": "skip",
                    "c1": random.randint(1,stage["e"]*input_channel),
                    "k": random.randint(1,int(stage["k"]/2))*2+1,
                    "c2": stage_conf["block"]["c2"],
                    "s": 1
                })
                config["stage"].append(stage_conf)
        else:
            print("Only linear space evaluated for random policy!")
        return config

    def step(self, loss, config_eval)-> None:
        if loss < self.opt_loss:
            self.opt_loss = loss
            self.opt_model = config_eval
        return
