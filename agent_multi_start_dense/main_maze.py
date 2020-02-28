#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Yandong
# Time  :2020/02/03


from DQN_agent import QAgent
from maze_env import maze_grid
import random
import logging
import sys

logging.basicConfig(
    # 日志级别
    level=logging.INFO,
    # 日志格式
    # 时间、代码所在文件名、代码行号、日志级别名字、日志信息
    format="%(levelname)-8s: %(asctime)s: %(message)s",
    # 打印日志的时间
    datefmt='%a, %d %b %Y %H:%M:%S',
    # 日志文件存放的目录（目录必须存在）及日志文件名
    # filename = 'd:/python_log/report.log',
    stream=sys.stdout,
    # 打开日志文件的方式
    filemode='w')


def train_agent(starts, end, n_height, n_width, obstacles, model_file=None):
    """
    build the environment
    @param starts: list
    @param end: tuple
    @param n_height: int
    @param n_width: int
    @param obstacles: list
    @param model_file: str model weights

    """
    env = maze_grid(starts, end, n_height, n_width, obstacles)  # maze
    env.render()
    # input('show')
    # q learning model
    agents = QAgent(env,
                    memory_capacity=50 * env.maze_size,    # experience memory
                    model_file=model_file)
    logging.info("Learning...")
    agents.learning(max_episodes=1000,
                    batch_size=512,
                    gamma=0.95,
                    min_epsilon=0.01)


def generate_obstacles(starts, end, seed_num, grid_num=100, percent=0.2):
    """
    generate the obstacles
    @param starts: the agent start list
    @param end: goal
    @param seed_num: random obstacle
    @param grid_num: maze size
    @param percent: obstacles in maze
    @return: the position of obstacles
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
    # print('obstacles:', obs_pos)
    return list(set(obs_pos))


if __name__ == "__main__":
    # choose the different instance to choose
    model_file = None  # add the weights files

    # experiment 1
    # starts = [(0, 2), (2, 8), (8, 1), (9, 9)]
    # end = (4, 4)
    # seed_num = 1

    # experiment 2
    # starts = [(2, 2), (7, 2), (1, 7), (7, 7)]
    # end = (4, 4)
    # seed_num = 6

    # experiment 3
    starts = [(0, 0), (8, 1), (0, 8), (8, 9)]
    end = (4, 4)
    seed_num = 3

    maze_length, maze_width = 10, 10
    obstacles = generate_obstacles(
        starts,
        end,
        seed_num,
        grid_num=maze_length *
        maze_width,
        percent=0.2)
    train_agent(starts, end, maze_length, maze_width, obstacles, model_file)
