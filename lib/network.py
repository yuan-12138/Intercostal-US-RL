import torch
import torch.nn as nn
import torch.functional as F
import torch.optim as optim
import numpy as np
import time
import sys
import spconv.pytorch as spconv
from gymnasium.spaces import Discrete, Box, Tuple

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class DeepQNetwork(nn.Module):
    def __init__(self, observation_space: Box, 
                 action_space: Discrete,
                 stack_num: int = 1
                 ):
        super(DeepQNetwork, self).__init__()
        self.box_input_dims = observation_space.shape
        self.n_actions = action_space.n
        self.stack_num = stack_num
        self.conv4 = nn.Sequential(
            nn.Conv3d(self.box_input_dims[1], 32, kernel_size=3, stride=1),
            nn.ReLU()
        )
        self.conv4_sp = spconv.SparseSequential(
            spconv.SparseConv3d(3, 32, kernel_size=3, stride=1),
            nn.ReLU()
        )
        self.max_pool2 = nn.MaxPool3d(2, 2)
        self.max_pool2_sp = spconv.SparseMaxPool3d(2, 2)
        self.conv5 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        self.conv5_sp = spconv.SparseSequential(
            spconv.SparseConv3d(32, 64, kernel_size=3, stride=1, indice_key="subm0"),
            nn.ReLU()
        )
        self.conv6 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=3, stride=1),
            nn.ReLU()
        )
        self.conv6_sp = spconv.SparseSequential(
            spconv.SparseConv3d(64, 128, kernel_size=3, stride=1),
            nn.ReLU()
        )
        # self.output_dims = self.get_output_shape_img(self.img_input_dims) + self.get_output_shape_box(self.box_input_dims)
        self.output_dims = 1024
        self.fc1_a = nn.Sequential(
            nn.Linear(self.output_dims, 256),
            nn.ReLU()
        )
        self.fc1_v = nn.Sequential(
            nn.Linear(self.output_dims, 256),
            nn.ReLU()
        )
        self.fc2_a = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU()
        )
        self.fc2_v = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU()
        )
        self.fc3_a = nn.Sequential(
            nn.Linear(64, self.n_actions)
        )
        self.fc3_v = nn.Sequential(
            nn.Linear(64, 1)
        )
        self.feature_net = nn.Sequential(
            self.conv4,
            self.max_pool2,
            self.conv5,
            self.max_pool2,
            self.conv6,
            self.max_pool2,
            nn.Flatten()
        )
        self.a_net = nn.Sequential(
            self.fc1_a,
            self.fc2_a,
            self.fc3_a
        )
        self.v_net = nn.Sequential(
            self.fc1_v,
            self.fc2_v,
            self.fc3_v
        )

    def forward1(self, obs, state=None, info={}):
        y = obs
        y = torch.tensor(y, dtype=torch.float32).to(device)
        y = spconv.SparseConvTensor.from_dense(y)
        y = self.conv4_sp(y)
        y = self.max_pool2_sp(y)
        y = self.conv5_sp(y)
        y = self.max_pool2_sp(y)
        y = self.conv6_sp(y)
        y = self.max_pool2_sp(y)
        y = y.dense()
        # print(y.shape)
        out = y.contiguous().view(-1, 1024)
        # print(box_out.shape)
        # out = torch.cat((img_out, box_out), dim = 1)
        # out = box_out
        # print(out.shape)
        advantages = self.fc1_a(out)
        advantages = self.fc2_a(advantages)
        advantages = self.fc3_a(advantages)
        value = self.fc1_v(out)
        value = self.fc2_v(value)
        value = self.fc3_v(value)
        q_vals = value + (advantages - advantages.mean())
        # print(q_vals.shape)
        return q_vals, state

    def forward(self, obs, state=None, info={}):
        y = obs
        # Stacked frames
        if len(y.shape) == 6:
            y = y.reshape((-1, 9, 30, 30, 30))
        # x = torch.tensor(x, dtype=torch.float32).to(device)
        y = torch.tensor(y, dtype=torch.float32).to(device)
        out = self.feature_net(y)
        advantages = self.a_net(out)
        value = self.v_net(out)
        q_vals = value + (advantages - advantages.mean())
        return q_vals, state
    
class RelayBuffer:
    def __init__(self, size):
        self.storage = []
        self.size = size
        self.next_idx = 0
    
    def __len__(self):
        return len(self.storage)
    
    def add(self, state, action, reward, next_action, done):
        data = (state, action, reward, next_action, done)
        if self.next_idx >= len(self.storage):
            self.storage.append(data)
        else:
            self.storage[int(self.next_idx)] = data
        self.next_idx = int((int(self.next_idx) + 1) % self.size)
    
    def encode_sample(self, indices):
        imgs, boxes, actions, rewards, next_imgs, next_boxes, dones = [], [], [], [], [], [], []
        for i in indices:
            data = self.storage[i]
            state, action, reward, next_state, done = data
            imgs.append(np.array(state[0], copy=False))
            boxes.append(np.array(state[1], copy=False))
            actions.append(action)
            rewards.append(reward)
            next_imgs.append(np.array(next_state[0], copy=False))
            next_boxes.append(np.array(next_state[1], copy=False))
            dones.append(done)
        return np.array(imgs), np.array(boxes), np.array(actions), \
            np.array(rewards), np.array(next_imgs), np.array(next_boxes),\
            np.array(dones)
    
    def sample(self, batch_size):
        indices = np.random.randint(0, len(self.storage) - 1, size=batch_size)
        return self.encode_sample(indices)
    
class PrioritizedRelayBuffer(RelayBuffer):
    def __init__(self, size, alpha):
        super(PrioritizedRelayBuffer, self).__init__(size)
        self.max_priority, self.tree_ptr = 1.0, 0
        self.alpha = alpha
        capacity = 1
        while capacity < self.size:
            capacity *= 2
        self.sum_tree = segment_tree.SumSegmentTree(capacity)
        self.min_tree = segment_tree.MinSegmentTree(capacity)
    
    def add(self, state, action, reward, next_action, done):
        super().add(state, action, reward, next_action, done)
        self.sum_tree[int(self.tree_ptr)] = self.max_priority ** self.alpha
        self.min_tree[int(self.tree_ptr)] = self.max_priority ** self.alpha
        self.tree_ptr = int((int(self.tree_ptr) + 1) % self.size)
    
    def sample(self, batch_size, beta):
        indices = self.sample_proportional(batch_size)
        samples = self.encode_sample(indices)
        weights = np.array([self.calculate_weight(i, beta) for i in indices])
        return samples, indices, weights
    
    def calculate_weight(self, idx, beta):
        p_min = self.min_tree.min() / self.sum_tree.sum()
        max_weight = (p_min * len(self)) ** (-beta)
        p_sample = self.sum_tree[idx] / self.sum_tree.sum()
        weight = (p_sample * len(self)) ** (-beta)
        weight = weight / max_weight
        return weight

    def update_priorities(self, indices, priorities):
        for idx, p in zip(indices, priorities):
            self.sum_tree[idx] = p ** self.alpha
            self.min_tree[idx] = p ** self.alpha
            self.max_priority = max(self.max_priority, p)
    
    def sample_proportional(self, batch_size):
        indices = []
        p_total = self.sum_tree.sum(0, len(self)-1)
        segment = p_total / batch_size
        for i in range(batch_size):
            a = i * segment
            b = (i+1) * segment
            ub = np.random.uniform(a, b)
            idx = self.sum_tree.retrieve(ub)
            indices.append(idx)
        return indices

if __name__ == '__main__':
    r = PrioritizedRelayBuffer(100, 0.6)
    r.add([1, 0], 2, 3, [4, 0], 5)
    r.add([1, 0], 2, 3, [4, 0], 6)
    r.add([1, 0], 2, 3, [4, 0], 7)
    data, indices, weights = r.sample(1, 0.4)
    print(weights[0])
    print(data)
    r.update_priorities(indices, [3])
    a, b, c, d, e, f, g = data
    del a
    del b
    del c
    del d
    del e
    del f
    del g
    print(r.storage)
    print(r.sum_tree[indices[0]])
    data, indices, weights = r.sample(1, 0.4)
    print(data)
    print(r.storage)

