#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Yandong
# Time  :2020/02/03


from DQN_agent import QAgent
from maze_env import maze_grid
import random


def train_agent(starts, end, n_height, n_width, obstacles, model_file=None):
    # build the environment
    env = maze_grid(starts, end, n_height, n_width, obstacles)  # maze
    env.render()
    # input('123')
    # q learning model
    agents = QAgent(env,
                    memory_capacity=100*env.maze_size,    # experience memory
                    model_file=model_file)
    print("Learning...")
    agents.learning(max_episodes=1000,
                    batch_size=512,
                    gamma=0.95,
                    min_epsilon=0.01)

def generate_obstacles(starts, end, seed_num, grid_num=100, percent=0.2):
    """
    generate the obstacles
    """
    random.seed(seed_num)
    obs_num = round(grid_num * percent)
    obs_pos = list()
    while len(obs_pos) != obs_num:
        pos = (random.randint(0, 9), random.randint(0, 9))
        if pos in starts or pos == end:
            continue
        else:
            obs_pos.append(pos)
    print('obstacles:', obs_pos)
    return obs_pos


if __name__ == "__main__":
    # experiment 1
    # starts = [(0, 2), (2, 8), (8, 1), (9, 9)]
    starts = [(2, 8)]
    end = (4, 4)
    seed_num = 1
    model_file = None

    maze_length, maze_width = 10, 10
    obstacles = generate_obstacles(starts, end, seed_num, grid_num=maze_length*maze_width, percent=0.2)
    train_agent(starts, end, maze_length, maze_width, obstacles, model_file)
