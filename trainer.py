import tianshou as ts
import datetime
import random
import numpy as np
import gymnasium
import time
import torch
import torch.nn.functional as F
import argparse
#import wandb
import sys, os
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR
from tianshou.data.batch import Batch
import torchvision
import wandb
# from tianshou.utils.net.discrete import IntrinsicCuriosityModule
from tianshou.policy.modelbased.icm import ICMPolicy

from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from tianshou.data import Batch, to_torch
from tianshou.utils.net.common import MLP

# from tianshou.utils import TensorboardLogger
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

# CHECKPOINT_PATH = None
# hdf5_path = None
# CHECKPOINT_PATH = "experiment/icm_new_action_fix/checkpoint.pth"
# # hdf5_path = "experiment/random_2/checkpoint.hdf5"
# log_path = "experiment/icm_new_action_fix_2"

from collections import deque

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

class CollectorProcessor:
    def __init__(self, size=100):
        self.episode_log = None
        self.main_log = deque(maxlen=size)
        self.main_log.append(0)
        self.baseline = 0
        self.episode_cnt = 0

    def preprocess_fn(self, **kwargs):
        """change reward to zero mean"""
        # if obs && env_id exist -> reset
        # if obs_next/act/rew/done/policy/env_id exist -> normal step
        if 'rew' not in kwargs:
            # means that it is called after env.reset(), it can only process the obs
            return Batch()  # none of the variables are needed to be updated
        else:
            n = len(kwargs['rew'])  # the number of envs in collector
            if self.episode_log is None:
                self.episode_log = [[] for i in range(n)]
            for i in range(n):
                self.episode_log[i].append(kwargs['rew'][i])
                # kwargs['rew'][i] -= self.baseline
            for i in range(n):
                if kwargs['done'][i]:
                    self.main_log.append(np.sum(self.episode_log[i]))
                    self.episode_log[i] = []
                    self.baseline = np.mean(self.main_log)
                    # self.writer.add_scalar('train_reward', self.main_log[-1], self.episode_cnt)
                    # self.writer.add_scalar('mean_reward', self.baseline, self.episode_cnt)
                    success_rate = np.sum(np.array(self.main_log) >= 10) / len(self.main_log)
                    wandb.log({'train_reward': self.main_log[-1], 'mean_reward': self.baseline, 'episode': self.episode_cnt, 'success_rate': success_rate})
                    self.episode_cnt += 1
            return Batch()

def run(seed, 
        replay_buffer_size, 
        learning_rate, 
        num_steps, 
        exp_steps,
        batch_size, 
        discount, 
        warming_up_steps, 
        start_step, 
        train_freq, 
        test_freq, 
        target_update_freq, 
        checkpoint_freq, 
        eps_train, 
        eps_train_final, 
        alpha, 
        beta_train, 
        beta_final, 
        envs_num, 
        log_path, 
        checkpoint_path, 
        hdf5_path,
        lr_step_size,
        lr_decay_rate,
        use_icm,
        th,
        w_dis,
        w_percentage,
        tumor_size):
    torch.cuda.empty_cache()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    # '''
    # Hyper Parameters
    # '''

    # seed = 40
    # relay_buffer_size = 4e3 #
    # learning_rate = 3e-5 #
    # start_step = 2e6
    # num_steps = 5e6 
    # exp_steps = 3e6
    # batch_size = 128 #
    # discount = 0.99
    # warming_up_steps = 4e3
    # train_freq = 1
    # test_freq = 8e3
    # target_update_freq = 5e3 #
    # checkpoint_freq = 1e4
    # eps_train = 0.99
    # eps_train_final = 0.05
    # print_freq = 1
    # decay_rate = 0.01
    # prior_eps = 1e-6
    # alpha = 0.6
    # beta_train = 0.4
    # beta_final = 1
    # envs_num = 16

    # Init wandb logger
    # Initialize name with date and time
    name = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    wandb.init(project="RobUS", name=name, config={
        "seed": seed,
        "replay_buffer_size": replay_buffer_size,
        "start_step": start_step,
        "num_steps": num_steps,
        "batch_size": batch_size,
        "discount": discount,
        "warming_up_steps": warming_up_steps,
        "train_freq": train_freq,
        "test_freq": test_freq,
        "learning_rate": learning_rate,
        "target_update_freq": target_update_freq,
        "checkpoint_freq": checkpoint_freq,
        "eps_train": eps_train,
        "eps_train_final": eps_train_final,
        "alpha": alpha,
        "beta_train": beta_train,
        "beta_final": beta_final,
        "envs_num": envs_num,
        "log_path": log_path,  
        "checkpoint_path": checkpoint_path,
        "hdf5_path": hdf5_path,
        "lr_step_size": lr_step_size,
        "lr_decay_rate": lr_decay_rate,
        "use_icm": True
        })
    
    env = environ.RobusENV(th=th,
                           w_dis=w_dis,
                           w_percentage=w_percentage,
                           if_render=False,
                           tumor_size=tumor_size)
    train_envs = ts.env.SubprocVectorEnv([lambda: environ.RobusENV(th=th, w_dis=w_dis, w_percentage=w_percentage) for _ in range(envs_num)])
    train_envs.seed(seed)
    
    net = network.DeepQNetwork(env.observation_space, env.action_space, stack_num=3).to(device)

    optim = torch.optim.Adam(net.parameters(), lr = learning_rate)
    scheduler = StepLR(optim, 
                    step_size = lr_step_size, # Period of learning rate decay
                    gamma = lr_decay_rate
                    )
    policy = ts.policy.DQNPolicy(model=net, 
                                optim=optim, 
                                discount_factor=discount, 
                                target_update_freq=target_update_freq,
                                is_double=True, 
                                clip_loss_grad = True
                                ).to(device)
    
    feature_net = network.DeepQNetwork(env.observation_space, env.action_space, stack_num=4).to(device).feature_net
    icm_net = IntrinsicCuriosityModule(
            feature_net,
            1024,
            9,
            hidden_sizes=[1024],
            device=device
        )

    if use_icm:
        icm_optim = torch.optim.Adam(icm_net.parameters(), lr=learning_rate)

        icm_scheduler = StepLR(icm_optim, 
                    step_size = lr_step_size, # Period of learning rate decay
                    gamma = 0.8
                    )
        policy = ICMPolicy(policy, icm_net, icm_optim, 0.1, 0.1, 0.2).to(device)

    buffer = ts.data.PrioritizedVectorReplayBuffer(
                                             total_size=envs_num*replay_buffer_size, 
                                             buffer_num=envs_num,
                                             alpha=alpha, 
                                             beta=beta_train)
    if hdf5_path:
        print("Loading buffer")
        buffer.load_hdf5(hdf5_path, device)

    processor = CollectorProcessor(size=100)
    train_collector = ts.data.Collector(policy, train_envs, buffer, preprocess_fn=processor.preprocess_fn, exploration_noise=True)
    test_collector = ts.data.Collector(policy=policy, env=env, buffer=None, exploration_noise=False)
    if checkpoint_path:
        policy.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print("Loaded agent from: ", checkpoint_path)

    best_reward = 0
    best_length = 0

    if not hdf5_path:
        print("Pre-collecting")
        s = time.time()
        if start_step == 0: 
            train_collector.collect(n_step=warming_up_steps*envs_num, random=True)
        else:
            eps = eps_train - start_step / exp_steps * \
                (eps_train - eps_train_final)
            beta = beta_train - start_step / num_steps * \
                (beta_train - beta_final)
            eps = max(eps_train_final, eps)
            policy.set_eps(eps)
            buffer.set_beta(beta)
            train_collector.collect(n_step=warming_up_steps*envs_num, random=False)
        e = time.time()
        print(f"Pre-collecting time: {e-s}")

    print("Training")
    for i in range(int(start_step), int(num_steps), train_freq*envs_num):  # total step
       # Checkpoint
       
       #if i == 0 and not hdf5_path:
       #    hdf5_path = os.path.join(log_path, f"initial.hdf5")
       #     buffer.save_hdf5(hdf5_path)
        if i % checkpoint_freq == 0 and i != 0:
       #     hdf5_path = os.path.join(log_path, f"checkpoint.hdf5")
            ckpt_path = os.path.join(log_path, f"checkpoint.pth")
       #     buffer.save_hdf5(hdf5_path)
            torch.save(policy.state_dict(), ckpt_path)

        # Calculate eps and beta
        if i <= num_steps:
            eps = eps_train - i / exp_steps * \
                (eps_train - eps_train_final)
            beta = beta_train - i / num_steps * \
                    (beta_train - beta_final)
        else:
            eps = eps_train_final
            beta = beta_final
        eps = max(eps_train_final, eps)

        # Collect steps and set eps, beta
        policy.set_eps(eps)
        buffer.set_beta(beta)
        collect_result = train_collector.collect(n_step=train_freq*envs_num)

        # writer.add_scalar('train_reward', train_reward, i)

        # Test policy
        if i % test_freq == 0:
            test_collector.reset_env()
            result = test_collector.collect(n_episode=10)
            test_reward = result['rew']
            test_len = result['len']
            wandb.log({"test_reward": test_reward}, step=i)
            if test_reward > best_reward:
                best_reward = test_reward
                best_length = test_len
                torch.save(policy.state_dict(), os.path.join(log_path, "best_policy.pth"))
            print("********************************************************")
            print("Steps: {}".format(i))
            print("% Time spent exploring: {}".format(int(100 * eps)))
            print("Learning Rate: ", policy.optim.param_groups[-1]['lr'])
            print("Beta: ", beta)
            print("Test Reward: ", test_reward)
            print("Test Length: ", test_len)
            print("Best Reward: ", best_reward)
            print("Best Length: ", best_length)
            print("********************************************************")

        loss = policy.update(batch_size, train_collector.buffer)
        wandb.log({"loss": loss, "learning_rate": policy.optim.param_groups[-1]['lr']}, step=i)
        scheduler.step()
        if use_icm:
            icm_scheduler.step()

