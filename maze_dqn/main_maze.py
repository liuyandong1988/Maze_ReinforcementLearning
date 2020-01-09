#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Yandong
# Time  :2019/12/7 20:54


from DQN_agent import QAgent
from maze_env import maze_grid
from keras.models import load_model


def train_agent(start, ends, obstacles, load_model=False):

    env = maze_grid(start, ends, obstacles)  #maze
    env.render()
    agent = QAgent(env,
                   memory_capacity=100*env.maze_size,    # experience memory
                   hidden_dim=100, model_file=load_model)
    env.reset()
    print("Learning...")
    agent.learning(max_episodes=300,
                   batch_size=512,
                   gamma=0.9,
                   min_epsilon=0.01)


if __name__ == "__main__":
    # choose the different instance
    start = (2, 9)
    ends = [(5, 4)]  # many ends
    # instance 1
    # obstacles = [(4, 2), (4, 3), (4, 4), (4, 5), (4, 6), (1, 7), (2, 7), (3, 7), (4, 7), (6, 7), (7, 7), (8, 7)] #1

    # instance 2
    # obstacles = [(4, 2), (4, 3), (4, 4), (4, 5), (4, 6), (1, 7), (2, 7), (3, 7), (4, 7), (5, 7), (6, 7), (7, 7), (8, 7)] #2

    # # instance difficult
    start = (5, 9)
    ends = [(4, 5)]
    obstacles = [(2, 7), (3, 7), (4, 7), (5, 7), (6, 7), (7, 7), (3, 5), (6, 5), (2, 3), (3, 3), (6, 3), (7, 3)]
    model_path = './model/dqn_maze_300.h5'
    train_agent(start, ends, obstacles, False)
