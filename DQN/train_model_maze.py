import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from model import DQN
from maze import Maze

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
writer = SummaryWriter("./DQN/runs")
maze = Maze()
dqn = DQN(maze.n_features, maze.n_actions).to(device)

def run_maze():
    step = 0
    for episode in range(400):
        episode_step = 0
        observation = torch.FloatTensor(maze.reset()).to(device)
        while True:
            maze.render()
            action = dqn.act(observation)
            next_observation, reward, done = maze.step(action)
            next_observation = torch.FloatTensor(next_observation).to(device)
            dqn.store(observation, action, reward, next_observation)

            if step > 200 and step % 5 == 0:
                dqn.train()
                if step % 100 == 0:
                    torch.save(dqn.state_dict(), 'model_maze.pth')
                    writer.add_scalar('Loss/train', dqn.get_last_loss(), dqn.get_learn_step_counter())
                    writer.add_scalar('1-Exploration_rate/train', dqn.get_exploration_rate(), dqn.get_learn_step_counter())
                    writer.add_scalar('episode_step/train', episode_step, dqn.get_learn_step_counter())
                    print('episode: ', episode, '  step: ', step, '  loss:', dqn.get_last_loss())

            observation = next_observation
            step += 1
            episode_step += 1
            if done:
                break
    print('game over')
    torch.save(dqn.state_dict(), 'model_maze.pth')
    maze.destroy()

maze.after(500, run_maze)
maze.mainloop()


