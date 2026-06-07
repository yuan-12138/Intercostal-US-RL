import network
import torch
import sys
import numpy as np
from gymnasium.spaces import Discrete, Box, Tuple
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR

class DQNAgent():
    def __init__(self, 
                 observation_space: Tuple,
                 action_space: Discrete,
                 relay_buffer: network.PrioritizedRelayBuffer,
                 lr,
                 batch_size, 
                 gamma, 
                 beta,
                 prior_eps,
                 device
                 ):
        self.relay_buffer = relay_buffer
        self.batch_size = batch_size
        self.gamma = gamma
        self.lr = lr
        self.policy_network = network.DeepQNetwork(observation_space, 
                                                   action_space).to(device)
        self.target_network = network.DeepQNetwork(observation_space,
                                                    action_space).to(device)
        self.update_target_network()
        self.target_network.eval()
        # self.optimizer = torch.optim.RMSprop(self.policy_network.parameters(), lr = self.lr)
        self.optimizer = torch.optim.Adam(self.policy_network.parameters(), lr = self.lr)
        # self.optimizer = torch.optim.SGD(self.policy_network.parameters(), lr = self.lr)
        self.scheduler = StepLR(self.optimizer, 
                                step_size = 5e3, # Period of learning rate decay
                                gamma = 1)
        self.beta = beta
        self.prior_eps = prior_eps
        self.device = device
        self.loss = torch.Tensor([0.0]).to(device)

    def optimise_td_loss(self):
        """
        Optimise the TD-error over a single minibatch of transitions
        :return: the loss
        """
        device = self.device
        samples, indices, weights = self.relay_buffer.sample(self.batch_size, self.beta)
        weights = torch.tensor(weights.reshape(-1, 1), dtype=torch.float32).to(device)
        imgs, boxes, actions, rewards, next_imgs, next_boxes, dones = samples
        imgs = torch.tensor(imgs, dtype=torch.float32).to(device).squeeze(1)
        boxes = torch.tensor(boxes, dtype=torch.float32).to(device).squeeze(1)
        actions = torch.tensor(actions, dtype=torch.int64).to(device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        next_imgs = torch.tensor(next_imgs, dtype=torch.float32).to(device).squeeze(1)
        next_boxes = torch.tensor(next_boxes, dtype=torch.float32).to(device).squeeze(1)
        dones = torch.tensor(dones, dtype=torch.float32).to(device)

        with torch.no_grad():
            _, max_next_actions = self.policy_network(next_imgs, next_boxes).max(1)
            max_next_q_values = self.target_network(next_imgs, next_boxes).gather(1, max_next_actions.unsqueeze(1)).squeeze()
            target_q_values = rewards + (1-dones) * self.gamma * max_next_q_values
        
        eval_q_values = self.policy_network(imgs, boxes)
        eval_q_values = eval_q_values.gather(1, actions.unsqueeze(1)).squeeze()
        loss_elementwise = F.smooth_l1_loss(eval_q_values, target_q_values, reduction="none")
        loss = torch.mean(loss_elementwise * weights)

        with torch.no_grad():
            self.loss = loss
        self.optimizer.zero_grad()
        loss.backward()
        torch.set_printoptions(threshold=sys.maxsize)
        # print(self.policy_network.conv1[0].weight.grad)
        # print(self.policy_network.conv4[0].weight.grad)
        self.optimizer.step()

        loss_for_prior = loss_elementwise.detach().cpu().numpy()
        priorities = loss_for_prior + self.prior_eps
        self.relay_buffer.update_priorities(indices, priorities)

        del imgs
        del boxes
        del next_imgs
        del next_boxes
        return loss.item()

    def update_target_network(self):
        """
        Update the target Q-network by copying the weights from the current Q-network
        """
        self.target_network.load_state_dict(self.policy_network.state_dict())

    def greedy_action(self, img, box):
        device = self.device
        img = torch.tensor(img, dtype=torch.float32).to(device)
        box = torch.tensor(box, dtype=torch.float32).to(device)
        with torch.no_grad():
            q_values = self.policy_network(img, box)
            _, action = q_values.max(1)
            return action.item()

if __name__ == "__main__":
    dones = np.array([True, False])
    print(torch.tensor(dones, dtype=torch.float32))
