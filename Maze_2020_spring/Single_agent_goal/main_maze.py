#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Yandong
# Time  :2020/02/03


from DQN_agent import QAgent
from maze_env import maze_grid
import random


def train_agent(start, ends, n_height, n_width, obstacles, model_file=None):
    # build the environment
    env = maze_grid(start, ends, n_height, n_width, obstacles)  #maze
    env.render()
    # q learning model
    agent = QAgent(env,
                   memory_capacity=100*env.maze_size,    # experience memory
                   hidden_dim=256,
                   model_file=model_file)
    print("Learning...")
    agent.learning(max_episodes=3000,
                   batch_size=512,
                   gamma=0.9,
                   min_epsilon=0.01)


def generate_obstacles(start, goals, seed_num, grid_num=100, percent=0.2):
    """
    generate the obstacles
    """
    random.seed(seed_num)
    obs_num = grid_num * percent
    obs_pos = list()
    while len(obs_pos) != obs_num:
        pos = (random.randint(0, 9), random.randint(0, 9))
        if pos == start or pos in goals:
            continue
        else:
            obs_pos.append(pos)
    print('obstacles:', obs_pos)
    return obs_pos


if __name__ == "__main__":
    # choose the different instance
    start = (4, 4)
    ends = [(7, 2), (4, 0), (0, 9), (9, 8), (5, 7)]  # many ends
    # ends = [(0, 9), (9, 2)]
    maze_length = 10
    maze_width = 10
    seed_num = 1
    model_file = None
    obstacles = generate_obstacles(start, ends, seed_num, grid_num=maze_length*maze_width, percent=0.2)
    train_agent(start, ends, maze_length, maze_width, obstacles, model_file)
