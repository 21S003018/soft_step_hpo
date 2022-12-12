from bayes_opt import BayesianOptimization
from const import *
import json
import random


class BayesPolicy():
    def __init__(self, search_space: str, observe) -> None:
        self.search_space = search_space
        with open(self.search_space, "r") as f:
            self.config = json.load(f)
        self.opt_model = None
        self.opt_loss = 1e6
        self.tag = "bayes"
        self.observe = observe
        self.iter = 0
        self.controller = BayesianOptimization(self.objective, {
            conv_in_channel: (1, 32.99),
            block_in_conv1_channel: (1, 64.99),
            block_in_conv2_kernel: (1, 3.99),
            block_in_conv3_channel: (1, 32.99),
            stages_0_block_conv1_channel: (1, 192.99),
            stages_0_block_conv2_kernel: (1, 3.99),
            stages_0_block_conv3_channel: (1, 32.99),
            stages_0_skips_0_conv1_channel: (1, 192.99),
            stages_0_skips_0_conv2_kernel: (1, 3.99),
            stages_0_skips_1_conv1_channel: (1, 192.99),
            stages_0_skips_1_conv2_kernel: (1, 3.99),
            stages_1_block_conv1_channel: (1, 192.99),
            stages_1_block_conv2_kernel: (1, 3.99),
            stages_1_block_conv3_channel: (1, 64.99),
            stages_1_skips_0_conv1_channel: (1, 384.99),
            stages_1_skips_0_conv2_kernel: (1, 3.99),
            stages_1_skips_1_conv1_channel: (1, 384.99),
            stages_1_skips_1_conv2_kernel: (1, 3.99),
            stages_2_block_conv1_channel: (1, 384.99),
            stages_2_block_conv2_kernel: (1, 3.99),
            stages_2_block_conv3_channel: (1, 96.99),
            stages_2_skips_0_conv1_channel: (1, 576.99),
            stages_2_skips_0_conv2_kernel: (1, 3.99),
            stages_2_skips_1_conv1_channel: (1, 576.99),
            stages_2_skips_1_conv2_kernel: (1, 3.99),
            stages_3_block_conv1_channel: (1, 576.99),
            stages_3_block_conv2_kernel: (1, 3.99),
            stages_3_block_conv3_channel: (1, 128.99),
            stages_3_skips_0_conv1_channel: (1, 768.99),
            stages_3_skips_0_conv2_kernel: (1, 3.99),
            stages_3_skips_1_conv1_channel: (1, 768.99),
            stages_3_skips_1_conv2_kernel: (1, 3.99),
            stages_3_skips_2_conv1_channel: (1, 768.99),
            stages_3_skips_2_conv2_kernel: (1, 3.99),
            stages_4_block_conv1_channel: (1, 768.99),
            stages_4_block_conv2_kernel: (1, 3.99),
            stages_4_block_conv3_channel: (1, 256.99),
            stages_4_skips_0_conv1_channel: (1, 1536.99),
            stages_4_skips_0_conv2_kernel: (1, 3.99),
            stages_4_skips_1_conv1_channel: (1, 1536.99),
            stages_4_skips_1_conv2_kernel: (1, 3.99),
            stages_4_skips_2_conv1_channel: (1, 1536.99),
            stages_4_skips_2_conv2_kernel: (1, 3.99),
            conv_out_channel: (1, 1024.99),
        },verbose=1)
        pass

    def objective(self, conv_in_channel, block_in_conv1_channel, block_in_conv2_kernel, block_in_conv3_channel, stages_0_block_conv1_channel, stages_0_block_conv2_kernel, stages_0_block_conv3_channel, stages_0_skips_0_conv1_channel, stages_0_skips_0_conv2_kernel, stages_0_skips_1_conv1_channel, stages_0_skips_1_conv2_kernel, stages_1_block_conv1_channel, stages_1_block_conv2_kernel, stages_1_block_conv3_channel, stages_1_skips_0_conv1_channel, stages_1_skips_0_conv2_kernel, stages_1_skips_1_conv1_channel, stages_1_skips_1_conv2_kernel, stages_2_block_conv1_channel, stages_2_block_conv2_kernel, stages_2_block_conv3_channel, stages_2_skips_0_conv1_channel, stages_2_skips_0_conv2_kernel, stages_2_skips_1_conv1_channel, stages_2_skips_1_conv2_kernel, stages_3_block_conv1_channel, stages_3_block_conv2_kernel, stages_3_block_conv3_channel, stages_3_skips_0_conv1_channel, stages_3_skips_0_conv2_kernel, stages_3_skips_1_conv1_channel, stages_3_skips_1_conv2_kernel, stages_3_skips_2_conv1_channel, stages_3_skips_2_conv2_kernel, stages_4_block_conv1_channel, stages_4_block_conv2_kernel, stages_4_block_conv3_channel, stages_4_skips_0_conv1_channel, stages_4_skips_0_conv2_kernel, stages_4_skips_1_conv1_channel, stages_4_skips_1_conv2_kernel, stages_4_skips_2_conv1_channel, stages_4_skips_2_conv2_kernel, conv_out_channel):
        config = {"type": self.config["type"]}
        config = {
            "type": "linear",
            "layer": {
                "conv_in": int(conv_in_channel),
                "conv_out": int(conv_out_channel)
            },
            "block": {
                "type": "normal",
                "c1": int(block_in_conv1_channel),
                "k": int(block_in_conv2_kernel)*2+1,
                "c2": int(block_in_conv3_channel),
                "s": 1
            },
            "stage": [
                {
                    "block": {
                        "type": "reduction",
                        "c1": int(stages_0_block_conv1_channel),
                        "k": int(stages_0_block_conv2_kernel)*2+1,
                        "c2": int(stages_0_block_conv3_channel),
                        "s": 2
                    },
                    "skip": [
                        {
                            "type": "skip",
                            "c1": int(stages_0_skips_0_conv1_channel),
                            "k": int(stages_0_skips_0_conv2_kernel)*2+1,
                            "c2": int(stages_0_block_conv3_channel),
                            "s": 1
                        },
                        {
                            "type": "skip",
                            "c1": int(stages_0_skips_1_conv1_channel),
                            "k": int(stages_0_skips_1_conv2_kernel)*2+1,
                            "c2": int(stages_0_block_conv3_channel),
                            "s": 1
                        }
                    ]
                },
                {
                    "block": {
                        "type": "reduction",
                        "c1": int(stages_1_block_conv1_channel),
                        "k": int(stages_1_block_conv2_kernel)*2+1,
                        "c2": int(stages_1_block_conv3_channel),
                        "s": 2
                    },
                    "skip": [
                        {
                            "type": "skip",
                            "c1": int(stages_1_skips_0_conv1_channel),
                            "k": int(stages_1_skips_0_conv2_kernel)*2+1,
                            "c2": int(stages_1_block_conv3_channel),
                            "s": 1
                        },
                        {
                            "type": "skip",
                            "c1": int(stages_1_skips_1_conv1_channel),
                            "k": int(stages_1_skips_1_conv2_kernel)*2+1,
                            "c2": int(stages_1_block_conv3_channel),
                            "s": 1
                        }
                    ]
                },
                {
                    "block": {
                        "type": "reduction",
                        "c1": int(stages_2_block_conv1_channel),
                        "k": int(stages_2_block_conv2_kernel)*2+1,
                        "c2": int(stages_2_block_conv3_channel),
                        "s": 2
                    },
                    "skip": [
                        {
                            "type": "skip",
                            "c1": int(stages_2_skips_0_conv1_channel),
                            "k": int(stages_2_skips_0_conv2_kernel)*2+1,
                            "c2": int(stages_2_block_conv3_channel),
                            "s": 1
                        },
                        {
                            "type": "skip",
                            "c1": int(stages_2_skips_1_conv1_channel),
                            "k": int(stages_2_skips_1_conv2_kernel)*2+1,
                            "c2": int(stages_2_block_conv3_channel),
                            "s": 1
                        }
                    ]
                },
                {
                    "block": {
                        "type": "normal",
                        "c1": int(stages_3_block_conv1_channel),
                        "k": int(stages_3_block_conv2_kernel)*2+1,
                        "c2": int(stages_3_block_conv3_channel),
                        "s": 1
                    },
                    "skip": [
                        {
                            "type": "skip",
                            "c1": int(stages_3_skips_0_conv1_channel),
                            "k": int(stages_3_skips_0_conv2_kernel)*2+1,
                            "c2": int(stages_3_block_conv3_channel),
                            "s": 1
                        },
                        {
                            "type": "skip",
                            "c1": int(stages_3_skips_1_conv1_channel),
                            "k": int(stages_3_skips_1_conv2_kernel)*2+1,
                            "c2": int(stages_3_block_conv3_channel),
                            "s": 1
                        },
                        {
                            "type": "skip",
                            "c1": int(stages_3_skips_2_conv1_channel),
                            "k": int(stages_3_skips_2_conv2_kernel)*2+1,
                            "c2": int(stages_3_block_conv3_channel),
                            "s": 1
                        }
                    ]
                },
                {
                    "block": {
                        "type": "normal",
                        "c1": int(stages_4_block_conv1_channel),
                        "k": int(stages_4_block_conv2_kernel)*2+1,
                        "c2": int(stages_4_block_conv3_channel),
                        "s": 1
                    },
                    "skip": [
                        {
                            "type": "skip",
                            "c1": int(stages_4_skips_0_conv1_channel),
                            "k": int(stages_4_skips_0_conv2_kernel)*2+1,
                            "c2": int(stages_4_block_conv3_channel),
                            "s": 1
                        },
                        {
                            "type": "skip",
                            "c1": int(stages_4_skips_1_conv1_channel),
                            "k": int(stages_4_skips_1_conv2_kernel)*2+1,
                            "c2": int(stages_4_block_conv3_channel),
                            "s": 1
                        },
                        {
                            "type": "skip",
                            "c1": int(stages_4_skips_2_conv1_channel),
                            "k": int(stages_4_skips_2_conv2_kernel)*2+1,
                            "c2": int(stages_4_block_conv3_channel),
                            "s": 1
                        }
                    ]
                }
            ]
        }
        self.iter += 1
        train_loss = self.observe(config)
        return 1-train_loss
