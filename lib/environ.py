import gymnasium
import gymnasium.spaces
from gymnasium.spaces import Discrete, Box, Tuple, Dict
from gymnasium.utils import seeding
import random
import enum
import numpy as np
#import numpysocket
import time, sys
from collections import deque
dir = "lib"
if not dir in sys.path:
    sys.path.append(dir)
import scene
import network
import torch

import importlib
importlib.reload(scene)
importlib.reload(network)

TCP_IP = ''
TCP_PORT = 9500
TUMOR_POINTS_NUM = 2025

class RobusENV(gymnasium.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self, 
                 mode = "train", 
                 if_render = False, 
                 video_id = 0, 
                 th = 3,
                 w_dis = 0.8,
                 w_percentage = 0.2,
                 tumor_size = 'all'):
        '''
        params：
        mode: train or test
        if_render: render or not
        video_id: folder id for saving videos, e.g. video_id=0 means saving videos in folder video_0
        th: threshold scanned bone number for a successful scan
        w_dis: weight for distance
        w_percentage: weight for percentage
        '''
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.blender_scene = scene.Scene()
        self.frames = deque([], maxlen=3)
        self.action_space = Discrete(n=9)
        self.observation_space = Box(low=0, high=1, shape=(1, 9, 30, 30, 30), dtype=np.float32) 
        self.mode = mode
        self.if_render = if_render
        self.video_id = video_id
        self.th = th
        self.w_dis = w_dis
        self.w_percentage = w_percentage
        self.rendered = False
        self.tumor_size = tumor_size

    def reset(self):
        self.blender_scene.reset(self.mode, self.tumor_size)
        obs = self.blender_scene.encode_obs()
        for _ in range(3):
            self.frames.append(obs)
        info = self._get_info()
        return self._get_ob(), info
    
    def _get_info(self):
        return {"Points left": self.blender_scene.get_points_left()}

    def step(self, action):
        reward, done = self.blender_scene.move_probe(action, 
                                                     th = self.th,
                                                     w_dis = self.w_dis,
                                                     w_percentage = self.w_percentage)
        obs = self.blender_scene.encode_obs(render=self.if_render, video_id=self.video_id)
        self.frames.append(obs)
        info = self._get_info()
        return self._get_ob(), reward, done, False, info
    
    def seed(self, seed):
        np.random.seed(seed)
        random.seed(seed)
    
    def _get_ob(self):
        # Concatenate the last three frames
        obs = np.concatenate(self.frames, axis=0)
        return obs

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    env = RobusENV()
    nn = network.DeepQNetwork(env.observation_space, env.action_space).to(device)
    # nn.get_output_shape_img(nn.img_input_dims)
    # nn.get_output_shape_box(nn.box_input_dims)
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, done, _, info = env.step(action)
        print(action)
        print(obs["3d"].shape)
        # img = torch.tensor(obs[0], dtype=torch.float32).to(device)
        # box = torch.tensor(obs[0], dtype=torch.float32).to(device)
        # img = torch.zeros(nn.img_input_dims).to(device)
        # box = torch.zeros(nn.box_input_dims).to(device)
        # print(img.shape)
        # print(box.shape)
        # print(reward, done, info)
        print(nn(obs))
        # nn.get_output_shape_img(nn.img_input_dims)
    # obs, info = env.reset()
    print(info)

    # env.reset()
    # env.conn.close()
