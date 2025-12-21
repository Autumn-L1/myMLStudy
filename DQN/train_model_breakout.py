import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo
import ale_py
gym.register_envs(ale_py)

from ConvDQNmodel import DQN
import gymEnvironment


#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
writer = SummaryWriter("./DQN/runs/breakout")
gymEnvironmentConfig = gymEnvironment.Config(
    env_name='ALE/Breakout-v5',
    display=True,
    screen_width=84,
    screen_height=84,
    action_repeat=4,
    random_start=30,
    #render_mode='human',
    memory_size = 4,
    record_video=True,
    video_folder="./DQN/record/Breakout-agent", 
    name_prefix="training",
    episode_trigger=lambda x: x % 50 == 0
)
frame_skip = 4

env = gymEnvironment.GymEnvironment(gymEnvironmentConfig)

dqn = DQN(env.action_size).to(device)

def run():
    step = 0
    for episode in range(200):
        episode_step = 0
        env.new_game()
        env.render()
        for _ in range(frame_skip):
            env._step(0)
        observation, _, _, _, _ = env.state
        observation = observation.float().to(device)
        while True:
            env.render()
            action = dqn.act(observation)
            for _ in range(frame_skip):
                env._step(action)
            next_observation, reward, done,  _, _ = env.state
            next_observation = next_observation.float().to(device)
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
    torch.save(dqn.state_dict(), 'model_breakout.pth')
run()
env.close()