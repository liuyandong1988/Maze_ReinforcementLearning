#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@File    : test_model.py
@Time    : 2020/2/25 19:47
@Author  : Yandong
@Function : Test the training model, success rate and optimal rate
"""
import DQN_agent as dqn
from maze_env import maze_grid
from main_maze import generate_obstacles
import matplotlib.pyplot as plt
import numpy as np
import time

def test_all_starts(starts, end, maze_length, maze_width, weight_file):
    """
    FIND the all start paths by training model
    @return: success_fail_list and step_list
    """
    success_fail_list, step_list = list(), list()
    # --- load the model dqn
    model = dqn.dqn_model((9, ), (25, ))
    # load the weight
    model.load_weights(weight_file)
    for start in starts:
        # --- env
        start_list_format = [start]
        env = maze_grid(start_list_format, end, maze_length, maze_width, obstacles)
        # env.render()
        # input('render')  # show the GUI
        # 1. find the each start path
        is_find, path_steps = find_path(start, model, env)
        success_fail_list.append(is_find)
        step_list.append(path_steps)
        del env  # delete the env
    # 2. calculate the success rate
    print(len(success_fail_list))
    success_rate = cal_success_rate(success_fail_list)
    print('Success rate: %03f' % success_rate)
    # 3. calculate the average path length (optimization rate)
    average_path_length = cal_average_length(step_list)
    print('Average steps: %02f' % average_path_length)
    # 4. show the path length
    show_steps(step_list)



def find_path(start, model, env, gui=False):
    """
    find the feasible path by model
    @param start:
    @param end:
    @param model:
    @param env:
    @return:
    """
    # --- observation the new states
    goal_states, agent_states = env.reset(start, test=None)  # get the initial states
    goal_states = np.reshape(goal_states, [9, ])[np.newaxis, :]
    agent_states = np.reshape(agent_states, [25, ])[np.newaxis, :]
    s0 = [goal_states, agent_states]
    # --- show the demo
    total_reward = 0
    is_done = False
    steps = 0
    while True:
        a0 = int(np.argmax(model.predict(s0)[0]))
        s1, r1, is_done, info = env.step(a0)
        if gui:
            env.render()
            # input('stop')
            time.sleep(0.02)
        steps += 1
        total_reward += r1
        if steps > 25:
            print("Fail reward: %02f"%total_reward)
            break
        if is_done:
            print('Arrive at the goal ...')
            break
        # update states
        s1 = np.reshape(s1, [25, ])
        s1 = s1[np.newaxis, :]
        s0 = [goal_states, s1]
    print('The reward: %d, the agent path distance is %d step.' % (total_reward, steps))
    return is_done, steps



def cal_success_rate(success_mark_list):
    """
    Calculate the success rate: success_path_number/total_paths
    @param success_mark_list:
    @return: success rate
    """
    total_paths = len(success_mark_list)
    succ_number = len([ i for i in success_mark_list if i is True])
    succ_rate =  succ_number/total_paths
    return succ_rate


def cal_average_length(paths):
    average_length = sum(paths) / len(paths)
    return average_length

def show_steps(steps):
    step_list = [ 0 if i is None else i for i in steps]
    plt.bar(range(len(step_list)), step_list)
    plt.xlabel('Start index')
    plt.ylabel('Path steps')
    plt.title('All path steps')
    plt.show()


def cal_starts(obstacles, row, col, except_start ):
    """
    calculate all starts
    @param obstacles:
    @param except_start:
    @return: starts
    """
    starts = list()
    for i in range(row):
        for j in range(col):
            start_pos = (i, j)
            if start_pos in obstacles or start_pos in except_start:
                continue
            else:
                starts.append(start_pos)
    print(len(starts))
    return starts





if __name__ == '__main__':
    """
    choose one experiment and test the find path success rate.
    """
    # experiment 1
    # weight_file = 'dense_model_exp1.h5'
    # # starts = [(0, 2), (2, 8), (8, 1), (9, 9)]
    # starts = [(8, 1)] # training start position
    # end = (4, 4)
    # except_start = [(9, 0)]
    # seed_num = 1


    # experiment 2
    # weight_file = 'dense_model_exp2.h5'
    # starts = [(2, 2), (7, 2), (1, 7), (7, 7)]
    # end = (4, 4)
    # seed_num = 6
    # except_start = list()

    # experiment 3
    # starts = [(0, 0), (8, 1), (0, 8), (8, 9)]
    starts = [(0, 0)]
    end = (4, 4)
    seed_num = 3
    except_start = [(4, 9)]
    weight_file = 'dense_model_exp3.h5'

    maze_length, maze_width = 10, 10
    obstacles = generate_obstacles(starts, end, seed_num, grid_num=maze_length * maze_width, percent=0.2)
    # except obstacles all starts and expect (9, 0)
    starts = cal_starts(obstacles, maze_length, maze_width, except_start)
    test_all_starts(starts, end, maze_length, maze_width, weight_file)
