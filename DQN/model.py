
import torch
import torch.nn as nn
import numpy as np
from collections import deque

class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size = 128):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
class DQN(nn.Module):
    def __init__(self, state_size, action_size, hidden_size = 128):
        super(DQN, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = 0.95 # 折扣因子
        self.epsilon = 1.0 # 探索率
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.memory_size = 100
        self.batch_size = 32
        self.n_step_update = 200
        self.eval_network = QNetwork(state_size, action_size, hidden_size)
        self.target_network = QNetwork(state_size, action_size, hidden_size)
        self.optimizer = torch.optim.SGD(self.eval_network.parameters(), lr=0.001)
        self.loss_fn = nn.MSELoss()
    
        self.memory = deque(maxlen=self.memory_size) # deque 存储数据，先进先出，超出最大长度时，会自动删除最旧的数据
        self.learn_step_counter = 0
        self.last_loss = 0.0
    def store(self, state, action, reward, next_state):
        #state = torch.FloatTensor(state)
        #next_state = torch.FloatTensor(next_state)
        self.memory.append((state, action, reward, next_state))
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_size)
        if not isinstance(state, torch.Tensor):
            raise Exception("state must be a tensor")
        state = state.unsqueeze(0)
        q_values = self.eval_network(state)
        return torch.argmax(q_values).item()
    
    def train(self):
        if len(self.memory) < self.batch_size:
            #raise Exception("Not enough memory")
            return
        indice = np.random.choice(len(self.memory), self.batch_size)
        batch = [self.memory[i] for i in indice]
        for state, action, reward, next_state in batch:
            # 计算Q值
            q_eval = self.eval_network(state)[action]
            # 计算目标Q值
            with torch.no_grad():
                q_next = torch.max(self.target_network(next_state))
                q_target = reward + self.gamma * q_next

            #计算loss
            loss = self.loss_fn(q_eval, q_target)
            self.last_loss = loss.item()
            #更新eval网络
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # 更新Q网络
        if (self.learn_step_counter+1) % self.n_step_update == 0:
            self.target_network.load_state_dict(self.eval_network.state_dict())
            print("更新Q网络, learn_step_counter=", self.learn_step_counter)
        self.learn_step_counter += 1

        # 更新探索率
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def get_last_loss(self):
        return self.last_loss
    
    def get_learn_step_counter(self):
        return self.learn_step_counter
    
    def get_exploration_rate(self):
        return self.epsilon
    
    def state_dict(self):
        return self.target_network.state_dict()
    

    

    def load_state_dict(self, state_dict):
        self.target_network.load_state_dict(state_dict)
        self.eval_network.load_state_dict(state_dict)
