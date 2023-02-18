import torch
from torch.nn import Conv2d, MaxPool2d, AvgPool2d, Flatten, Linear, PReLU, FractionalMaxPool2d, RNN
from torch import nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import json
import random


class PolicyNetwork(nn.Module):
    def __init__(self):
        super(PolicyNetwork, self).__init__()
        self.scale = nn.Sequential(
            Linear(44, 64),
            nn.Tanh(),
            Linear(64, 128),
            nn.Tanh()
        )
        self.rnn = RNN(input_size=128, hidden_size=128, num_layers=1)
        self.trans = []
        for _ in range(44):
            self.trans.append(nn.Sequential(
                Linear(128, 64),
                Linear(64, 3),
            ))
        self.trans = nn.Sequential(*self.trans)
        return

    def forward(self, x: torch.Tensor):
        x = self.scale(x)
        x = x.unsqueeze(0).repeat(44, 1, 1)
        outputs, _ = self.rnn(x)
        ret = []
        for i, output in enumerate(outputs):
            ret.append(self.trans[i](output))
        return ret


class ValueNetwork(nn.Module):
    def __init__(self):
        super(ValueNetwork, self).__init__()
        self.linear1 = nn.Linear(44, 64)
        self.linear2 = nn.Linear(64, 128)
        self.output = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.output(x)
        return x


class Environment():
    def __init__(self, search_space: str) -> None:
        self.search_space = search_space
        with open(self.search_space, "r") as f:
            self.config = json.load(f)
        pass

    def sample_state(self):
        state = []
        config = {"type": self.config["type"]}
        # layer
        config["layer"] = {
            "conv_in": random.randint(1, self.config["layer"]["conv_in"]),
            "conv_out": random.randint(1, self.config["layer"]["conv_out"]),
        }
        state.append(config["layer"]["conv_in"])
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
        state.append(config["block"]["c1"])
        state.append(config["block"]["k"])
        state.append(config["block"]["c2"])
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
            state.append(stage_conf["block"]["c1"])
            state.append(stage_conf["block"]["k"])
            state.append(stage_conf["block"]["c2"])
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
                state.append(stage_conf["skip"][-1]["c1"])
                state.append(stage_conf["skip"][-1]["k"])
            config["stage"].append(stage_conf)
        state.append(config["layer"]["conv_out"])
        return state

    def next_state(self, state, action):
        assert action.min() >= -1 and action.max() <= 1
        next_state = []
        config = {"type": self.config["type"]}
        # layer
        idx = 0
        config["layer"] = {
            "conv_in": state[idx]+action[idx] if (state[idx]+action[idx] >= 1 and state[idx]+action[idx] <= self.config["layer"]["conv_in"]) else state[idx],
            "conv_out": state[-1]+action[-1] if (state[-1]+action[-1] >= 1 and state[-1]+action[-1] <= self.config["layer"]["conv_out"]) else state[-1],
        }
        idx += 1
        next_state.append(config["layer"]["conv_in"])
        output_channel = self.config["layer"]["conv_in"]
        # block
        input_channel = output_channel
        config["block"] = {
            "type": self.config["block"]["type"],
            "c1": state[idx]+action[idx] if (state[idx]+action[idx] >= 1 and state[idx]+action[idx] <= self.config["block"]["e"]*input_channel) else state[idx],
            "k": state[idx+1]+action[idx+1]*2 if (state[idx+1]+action[idx+1]*2 >= 1 and state[idx+1]+action[idx+1]*2 <= self.config["block"]["k"]) else state[idx+1],
            "c2": state[idx+2]+action[idx+2] if (state[idx+2]+action[idx+2] >= 1 and state[idx+2]+action[idx+2] <= self.config["block"]["c"]) else state[idx+2],
            "s": self.config["block"]["s"]
        }
        idx += 3
        next_state.append(config["block"]["c1"])
        next_state.append(config["block"]["k"])
        next_state.append(config["block"]["c2"])
        output_channel = self.config["block"]["c"]
        # stage
        config["stage"] = []
        for i, stage in enumerate(self.config["stage"]):
            stage_conf = {}
            input_channel = output_channel
            stage_conf["block"] = {
                "type": stage["type"],
                "c1": state[idx] + action[idx] if (state[idx] + action[idx] >= 1 and state[idx] + action[idx] <= stage["e"]*input_channel) else state[idx],
                "k": state[idx+1] + action[idx+1]*2 if (state[idx+1] + action[idx+1]*2 >= 1 and state[idx+1] + action[idx+1]*2 <= stage["k"]) else state[idx+1],
                "c2": state[idx+2] + action[idx+2] if (state[idx+2] + action[idx+2] >= 1 and state[idx+2] + action[idx+2] <= stage["c"]) else state[idx+2],
                "s": stage["s"]
            }
            idx += 3
            next_state.append(stage_conf["block"]["c1"])
            next_state.append(stage_conf["block"]["k"])
            next_state.append(stage_conf["block"]["c2"])
            output_channel = stage["c"]
            stage_conf["skip"] = []
            input_channel = output_channel
            for _ in range(stage["n"]-1):
                stage_conf["skip"].append({
                    "type": "skip",
                    "c1": state[idx] + action[idx] if (state[idx] + action[idx] >= 1 and state[idx] + action[idx] <= stage["e"]*input_channel) else state[idx],
                    "k": state[idx+1] + action[idx+1]*2 if (state[idx+1] + action[idx+1]*2 >= 1 and state[idx+1] + action[idx+1]*2 <= stage["k"]) else state[idx+1],
                    "c2": stage_conf["block"]["c2"],
                    "s": 1
                })
                idx += 2
                next_state.append(stage_conf["skip"][-1]["c1"])
                next_state.append(stage_conf["skip"][-1]["k"])
            config["stage"].append(stage_conf)
        next_state.append(config["layer"]["conv_out"])
        return next_state

    def get_config(self, state):
        config = {"type": self.config["type"]}
        # layer
        idx = 0
        config["layer"] = {
            "conv_in": state[idx],
            "conv_out": state[-1],
        }
        idx += 1
        output_channel = self.config["layer"]["conv_in"]
        # block
        input_channel = output_channel
        config["block"] = {
            "type": self.config["block"]["type"],
            "c1": state[idx],
            "k": state[idx+1],
            "c2": state[idx+2],
            "s": self.config["block"]["s"]
        }
        idx += 3
        output_channel = self.config["block"]["c"]
        # stage
        config["stage"] = []
        for i, stage in enumerate(self.config["stage"]):
            stage_conf = {}
            input_channel = output_channel
            stage_conf["block"] = {
                "type": stage["type"],
                "c1": state[idx],
                "k": state[idx+1],
                "c2": state[idx+2],
                "s": stage["s"]
            }
            idx += 3
            output_channel = stage["c"]
            stage_conf["skip"] = []
            input_channel = output_channel
            for _ in range(stage["n"]-1):
                stage_conf["skip"].append({
                    "type": "skip",
                    "c1": state[idx],
                    "k": state[idx+1],
                    "c2": stage_conf["block"]["c2"],
                    "s": 1
                })
                idx += 2
            config["stage"].append(stage_conf)
        return config
