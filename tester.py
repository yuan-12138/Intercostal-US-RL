import tianshou as ts
import random
import numpy as np
import gymnasium
import torch
import torch.nn.functional as F
import argparse
#import wandb
import sys, os
# from tianshou.utils import TensorboardLogger
from tianshou.policy.modelbased.icm import ICMPolicy
dir = "lib"
if not dir in sys.path:
    sys.path.append(dir)
import network
import environ
import agent
import importlib
importlib.reload(network)
importlib.reload(environ)
importlib.reload(agent)

from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from tianshou.data import Batch, to_torch
from tianshou.utils.net.common import MLP

import yaml
try:
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
        RENDER = config['RENDER']
except:
    RENDER = True

if RENDER:
    import cv2
    import bpy

class IntrinsicCuriosityModule(nn.Module):
    """Implementation of Intrinsic Curiosity Module. arXiv:1705.05363.

    :param torch.nn.Module feature_net: a self-defined feature_net which output a
        flattened hidden state.
    :param int feature_dim: input dimension of the feature net.
    :param int action_dim: dimension of the action space.
    :param hidden_sizes: hidden layer sizes for forward and inverse models.
    :param device: device for the module.
    """

    def __init__(
        self,
        feature_net: nn.Module,
        feature_dim: int,
        action_dim: int,
        hidden_sizes: Sequence[int] = (),
        device: Union[str, torch.device] = "cpu"
    ) -> None:
        super().__init__()
        self.feature_net = feature_net
        self.forward_model = MLP(
            feature_dim + action_dim,
            output_dim=feature_dim,
            hidden_sizes=[1024],
            device=device
        )
        self.inverse_model = MLP(
            feature_dim * 2,
            output_dim=action_dim,
            hidden_sizes=[1024, 128],
            device=device
        )
        self.feature_dim = feature_dim
        self.action_dim = action_dim
        self.device = device

    def forward(
        self, s1: Union[np.ndarray, torch.Tensor],
        act: Union[np.ndarray, torch.Tensor], s2: Union[np.ndarray,
                                                        torch.Tensor], **kwargs: Any
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Mapping: s1, act, s2 -> mse_loss, act_hat."""
        s1 = to_torch(s1, dtype=torch.float32, device=self.device)
        s2 = to_torch(s2, dtype=torch.float32, device=self.device)
        phi1, phi2 = self.feature_net(s1), self.feature_net(s2)
        act = to_torch(act, dtype=torch.long, device=self.device)
        phi2_hat = self.forward_model(
            torch.cat([phi1, F.one_hot(act, num_classes=self.action_dim)], dim=1)
        )
        mse_loss = 0.5 * F.mse_loss(phi2_hat, phi2, reduction="none").sum(1)
        act_hat = self.inverse_model(torch.cat([phi1, phi2], dim=1))
        return mse_loss, act_hat

def run(seed,
        learning_rate,
        discount,
        target_update_freq,
        render,
        video_id,
        checkpoint_path,
        use_icm,
        th,
        w_dis,
        w_percentage,
        test_episodes,
        tumor_size
        ):
    torch.cuda.empty_cache()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    '''
    Hyper Parameters
    '''
#    seed = 40
#    learning_rate = 7e-5
#    discount = 0.99
#    target_update_freq = 5e2

    env = environ.RobusENV(mode='test', 
                           if_render=render, 
                           video_id=video_id, 
                           th=th,
                           w_dis=w_dis,
                           w_percentage=w_percentage,
                           tumor_size=tumor_size)
    env.seed(seed)
    # env = ts.env.DummyVectorEnv([lambda: env])
    
    torch.manual_seed(seed)
    net = network.DeepQNetwork(env.observation_space, env.action_space, stack_num=3).to(device)

    optim = torch.optim.Adam(net.parameters(), lr = learning_rate)
    policy = ts.policy.DQNPolicy(model=net, 
                                optim=optim, 
                                discount_factor=discount, 
                                target_update_freq=target_update_freq,
                                is_double=True, 
                                clip_loss_grad = True
                                ).to(device)
    if use_icm:
        feature_net = network.DeepQNetwork(env.observation_space, env.action_space, stack_num=4).to(device).feature_net
        icm_net = IntrinsicCuriosityModule(
                feature_net,
                1024,
                9,
                hidden_sizes=[1024],
                device=device
            )
        icm_optim = torch.optim.Adam(icm_net.parameters(), lr=learning_rate)
        policy = ICMPolicy(policy, icm_net, icm_optim, 0.1, 0.01,0.2).to(device)

    test_collector = ts.data.Collector(policy=policy, env=env, buffer=None, exploration_noise=False)
    if checkpoint_path:
        policy.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print("Loaded agent from: ", checkpoint_path)

    print("Testing")
    # Test policy
    success = 0
    vio = 0
    unfinished = 0
    steps = []
    percentages = []
    distances = []
    successes = []
    for i in range(test_episodes):
        if render:
            env.blender_scene.full_box.show()
            bpy.data.objects[env.blender_scene.bone_id].hide_render = False
        result = test_collector.collect(n_episode=1)
        test_reward = result['rew']
        test_len = result['len']

        if test_reward > 10 and test_len < 80:
            success += 1
            successes.append(1)
            steps.append(test_len)
            percentages.append(env.blender_scene.avg_percentage)
            distances.append(env.blender_scene.avg_dis)
            print("Success")
        elif test_len < 80:
            vio += 1
            successes.append(0)
            print("Violation")
        else:
            unfinished += 1
            successes.append(0)
            print("Unfinished")
        print(test_reward, test_len)

    print("Success Rate:", success, np.std(successes))
    print("Violation Rate", vio)
    print("Unifinished Rate", unfinished)
    print("Average Steps", np.mean(steps), np.std(steps))
    print("Average Percentage", np.mean(percentages), np.std(percentages))
    print("Average Distance", np.mean(distances)/150, np.std(distances)/150)

if __name__ == '__main__':
    run(seed=76,
        learning_rate=7e-5,
        discount=0.99,
        target_update_freq=5e2,
        render=True,
        video_id=76,
        checkpoint_path="D:/RobUS/experiment/beta_adjust_th_1/checkpoint.pth",
        use_icm=True,
        th=3,
        w_dis=0.8,
        w_percentage=0.2,
        test_episodes=1
        )
        
