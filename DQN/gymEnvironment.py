#reference https://github.com/devsisters/DQN-tensorflow/blob/master/dqn/environment.py
import gymnasium as gym
import random
import torch
import numpy as np
import torch
from PIL import Image
from collections import deque

class Config:
    def __init__(self,
                 env_name='CartPole-v1',
                 screen_width=84,
                 screen_height=84,
                 action_repeat=4,
                 random_start=10,
                 display=False,
                 render_mode='rgb_array',
                 memory_size = 4,
                 record_video=False,
                 video_folder="video_folder", 
                 name_prefix="training",
                 episode_trigger=lambda x: True):
      self.env_name = env_name
      self.screen_width = screen_width
      self.screen_height = screen_height
      self.action_repeat = action_repeat
      self.random_start = random_start
      self.display = display
      self.render_mode = render_mode
      self.memory_size = memory_size 
      self.record_video = record_video
      self.video_folder = video_folder
      self.name_prefix = name_prefix
      self.episode_trigger = episode_trigger

class Environment(object):
  def __init__(self, config):
    self.env = gym.make(config.env_name, render_mode = config.render_mode)

    screen_width, screen_height, self.action_repeat, self.random_start = \
        config.screen_width, config.screen_height, config.action_repeat, config.random_start

    self.display = config.display
    self.dims = (screen_width, screen_height)

    self._screen = None
    self.reward = 0
    self.terminal = True
    self.truncated = False
    self.info = {}
    self.memory = deque(maxlen=config.memory_size)
    self.memory_size = config.memory_size

    if config.record_video:
      self.env = gym.wrappers.RecordVideo(self.env, video_folder=config.video_folder, name_prefix=config.name_prefix, episode_trigger=config.episode_trigger)
      self.env = gym.wrappers.RecordEpisodeStatistics(self.env)

  def new_game(self, from_random_game=False):
    if self.lives == 0:
      self._screen = self.env.reset()
    self._step(1)
    self.render()
    return self.screen, 0, self.terminal, False, self.info  # obs, reward, terminated, truncated, info

  def new_random_game(self):
    self.new_game(True)
    for _ in range(random.randint(0, self.random_start - 1)):
      self._step(0)
    self.render()
    return self.screen, 0, self.terminal, False, self.info # obs, reward, terminated, truncated, info


  def _step(self, action):
    self._screen, self.reward, self.terminal, self.truncated,_ = self.env.step(action)
    self.memory.append(torch.from_numpy(np.array(self.screen)))
  def _random_step(self):
    action = self.env.action_space.sample()
    self._step(action)

  @ property
  def screen(self):
    im = Image.fromarray(self._screen)
    return im.resize(self.dims).convert('L')

  @property
  def action_size(self):
    return self.env.action_space.n

  @property
  def lives(self):
    #return self.env.ale.lives()
   return 0 if self.terminal else 1
  @property
  def state(self):
    if len(self.memory) ==  self.memory_size:
      screen_stacked = torch.stack(list(self.memory)).unsqueeze(0) 
      return screen_stacked, self.reward, self.terminal, self.truncated, self.info
    else:
      raise Exception('Not enough frames in memory')

  def render(self):
    if self.display:
      self.env.render()

  def after_act(self, action):
    self.render()

class GymEnvironment(Environment):
  def __init__(self, config):
    super(GymEnvironment, self).__init__(config)

  def act(self, action, is_training=True):
    cumulated_reward = 0
    start_lives = self.lives

    for _ in range(self.action_repeat):
      self._step(action)
      cumulated_reward = cumulated_reward + self.reward

      if is_training and start_lives > self.lives:
        cumulated_reward -= 1
        self.terminal = True

      if self.terminal:
        break

    self.reward = cumulated_reward

    self.after_act(action)
    return self.state
  def close(self):
    self.env.close()
  

class SimpleGymEnvironment(Environment):
  def __init__(self, config):
    super(SimpleGymEnvironment, self).__init__(config)

  def act(self, action, is_training=True):
    self._step(action)

    self.after_act(action)
    return self.state