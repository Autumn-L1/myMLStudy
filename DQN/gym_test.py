import gymnasium as gym
from gymnasium import envs
import ale_py
gym.register_envs(ale_py)

env_ids = [spec.id for spec in envs.registry.values()]
print('There are {0} envs in gym'.format(len(env_ids)))
print(env_ids)

import warnings
import time

# env = gym.make('ALE/Pong-v5', render_mode="human")
# # 环境初始化
# state = env.reset()
# # 循环交互
# while True:
#     # 渲染画面
#     env.render()
#     # 从动作空间随机获取一个动作
#     action = env.action_space.sample()
#     # agent与环境进行一步交互
#     state, reward, done, truncated, _ = env.step(action)
#     print('state = {0}; reward = {1}'.format(state, reward))
#     # 判断当前episode 是否完成
#     if done:
#         print('done')
#         break
#     #time.sleep(0.1)
# # 环境结束
# env.close()

import gymEnvironment
config = gymEnvironment.Config(
    env_name='ALE/Breakout-v5',
    display=True,
    screen_width=84,
    screen_height=84,
    action_repeat=4,
    random_start=30,
    #render_mode='human'
)

env = gymEnvironment.GymEnvironment(config)
state, reward, done, truncated, _ = env.new_game()
# 循环交互
while True:
    # 渲染画面
    env.render()
    # agent与环境进行一步交互
    env._random_step()
    state, reward, done, truncated, _ = env.state
    print('state = {0}; reward = {1}'.format(state, reward))
    # 判断当前episode 是否完成
    if done:
        print('done')
        break
    #time.sleep(0.1)
# 环境结束
env.close()