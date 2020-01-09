#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Yandong
# Time  :2019/12/22 15:02

import numpy as np
import random
import matplotlib.pyplot as plt
import math

GET_SMALL = False
GET_MIDDLE = False
GET_LARGE = False

state_index = {(False, False, False): 0,
               (False, False, True): 1,
               (False, True, False): 2,
               (False, True, True): 3,
               (True, False, False): 4,
               (True, False, True): 5,
               (True, True, False): 6,
               (True, True, True): 7}



action_space = 4
state_space = 48

Q_table = np.zeros((state_space, action_space))


def greed_policy(epsilon, state):
    if random.random() < epsilon:
        action = np.random.choice(action_space)
    else:
        action = np.argmax(Q_table[state])
    return action


def step(state, action):
    global GET_SMALL, GET_MIDDLE, GET_LARGE
    is_done = False
    reward = 0
    next_state = None
    state_bias = state_index[(GET_SMALL, GET_MIDDLE, GET_LARGE)]  # get the bias (get the cheese or not)
    mark_state = math.floor(state/8)
    # print('s', mark_state)
    # print('a', action)
    if action == 0:
        if mark_state % 3 == 0:
            next_state = mark_state
        else:
            next_state = mark_state - 1
            reward = get_reward(next_state)
    elif action == 1:
        if mark_state % 3 == 2:
            next_state = mark_state
        else:
            next_state = mark_state + 1
            reward = get_reward(next_state)
    elif action == 2:
        if mark_state < 3:
            next_state = mark_state
        else:
            next_state = mark_state - 3
            reward = get_reward(next_state)
    elif action == 3:
        if 3 <= mark_state < 6:
            next_state = mark_state
        else:
            next_state = mark_state + 3
            reward = get_reward(next_state)
    next_state = next_state*8 + state_bias
    reward -= 0.1  # the step cost

    if reward == -10.1:
        is_done = True
        # print('TRIP !!!')

    # if reward == 9.9:
    #     is_done = True
    #     # print('Big Cheese!!!')

    if GET_SMALL and GET_MIDDLE and GET_LARGE:
        is_done = True
        # print('All goals!!!')

    return next_state, reward, is_done


def get_reward(next_state):
    global GET_SMALL, GET_MIDDLE, GET_LARGE
    reward = 0
    if next_state == 0:
        reward = 0
    elif next_state == 1:
        if not GET_SMALL:
            reward = 1
            GET_SMALL = True
        else:
            pass
    elif next_state == 2:
        reward = 0
    elif next_state == 3:
        if not GET_MIDDLE:
            reward = 3
            GET_MIDDLE = True
        else:
            pass
    elif next_state == 4:
        reward = -10
    elif next_state == 5:
        if not GET_LARGE:
            reward = 10
            GET_LARGE = True
        else:
            pass
    return reward


def learning(episode):
    global GET_SMALL, GET_MIDDLE, GET_LARGE
    gamma = 0.99
    alpha = 0.99
    reward_list = list()
    for i in range(episode):
        total_reward = 0
        action_list = list()
        done = False
        s0 = 0
        # epsilon = np.max([1 / pow(i+1, 1/2), 0.01])
        epsilon = 1 / (i + 1)
        while not done:
            a0 = greed_policy(epsilon, s0)
            s1, reward, done = step(s0, a0)
            # update q value
            Q_table[s0, a0] = Q_table[s0, a0] + alpha*(reward + gamma*np.max(Q_table[s1]) - Q_table[s0, a0])
            total_reward += reward
            action_list.append(a0)
            s0 = s1
        GET_SMALL = False
        GET_MIDDLE = False
        GET_LARGE = False
        reward_list.append(total_reward)
        print('Episode: %d, total reward: %.01f, epsilon: %.02f'%(i, total_reward, epsilon))
        print('Action: %s'%show_action(action_list))

    print('Best reward: %.01f, action: %s'%(total_reward, show_action(action_list)))

    plt.plot(reward_list)
    plt.show()




def show_action(actions):
    trans = ['left', 'right', 'up', 'down']
    results = list()
    for i in actions:
        results.append(trans[int(i)])
    return results

if __name__ == '__main__':
    learning(300)





