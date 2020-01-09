#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Yandong
# Time  :2019/12/3 6:18

from gym import Env
from random import *
import matplotlib.pyplot as plt


class SarsaLambdaAgent(object):
    def __init__(self, env:Env):
        self.env = env
        self.Q = {}  # {s0:[,,,,,,],s1:[]} 数组内元素个数为行为空间大小
        self.E = {}  # Eligibility Trace
        self.state = None
        self._init_agent()
        return

    def _init_agent(self):
        self.state = self.env.reset()
        s_name = self._name_state(self.state)
        self._assert_state_in_QE(s_name, randomized=False)


    def _name_state(self, state):
        '''
        给个体的一个观测(状态）生成一个不重复的字符串作为Q、E字典里的键
        '''
        return str(state)

    def _assert_state_in_QE(self, s, randomized=True):
        if not self._is_state_in_Q(s):
            self._init_state_value(s, randomized)

    def _is_state_in_Q(self, s):
        return self.Q.get(s) is not None

    def _init_state_value(self, s_name, randomized = True):
        if not self._is_state_in_Q(s_name):
            self.Q[s_name], self.E[s_name] = {},{}
            for action in range(self.env.action_space.n):
                default_v = random() / 10 if randomized is True else 0.0
                self.Q[s_name][action] = default_v
                self.E[s_name][action] = 0.0

    def act(self, a):       # 执行一个行为
        return self.env.step(a)


    def performPolicy(self, s, episode_num, use_epsilon = True):
        epsilon = 1/(episode_num+1)
        Q_s = self.Q[s]
        str_act = None
        rand_value = random()
        action = None
        if use_epsilon and rand_value < epsilon:
            action = self.env.action_space.sample()
        else:
            str_act = max(Q_s, key=Q_s.get)
            action = int(str_act)
        return action, epsilon


    def learning(self, lambda_, gamma, alpha, max_episode_num):
        total_time = 0
        time_in_episode = 0
        num_episode = 1

        time_in_episode_history, total_step_history = list(), list()
        while num_episode <= max_episode_num:
            self._resetEValue()
            s0 = self._name_state(self.env.reset())
            a0, epsilon = self.performPolicy(s0, num_episode)
            # self.env.render()
            time_in_episode = 0
            is_done = False
            while not is_done:
                s1, r1, is_done, info = self.act(a0)
                self.env.render()
                s1 = self._name_state(s1)
                self._assert_state_in_QE(s1, randomized=True)
                a1, epsilon = self.performPolicy(s1, num_episode)
                q = self._get_(self.Q, s0, a0)
                q_prime = self._get_(self.Q, s1, a1)
                delta = r1 + gamma * q_prime - q
                e = self._get_(self.E, s0, a0)
                e = e + 1
                self._set_(self.E, s0, a0, e)  # set E before update E

                state_action_list = list(zip(self.E.keys(), self.E.values()))
                for s, a_es in state_action_list:
                    for a in range(self.env.action_space.n):
                        e_value = a_es[a]
                        old_q = self._get_(self.Q, s, a)
                        new_q = old_q + alpha * delta * e_value
                        new_e = gamma * lambda_ * e_value
                        self._set_(self.Q, s, a, new_q)
                        self._set_(self.E, s, a, new_e)

                # if num_episode == max_episode_num:
                #     print("t:{0:>2}: s:{1}, a:{2:10}, s1:{3}".
                #           format(time_in_episode, s0, a0, s1))

                s0, a0 = s1, a1
                time_in_episode += 1
            time_in_episode_history.append(time_in_episode)
            print("Episode {0} takes {1} steps epsilon:{2}.".format(
                num_episode, time_in_episode, epsilon))
            total_time += time_in_episode
            num_episode += 1
            total_step_history.append(total_time)


        plt.plot(total_step_history, time_in_episode_history)
        plt.xlabel('steps')
        plt.ylabel('running avg step')
        plt.show()

        # store the weight and score result
        with open('time_in_episode_history_2.txt', 'w') as f1:
            for i in time_in_episode_history:
                f1.write(str(i) + ' ')
        with open('total_step_history_2.txt', 'w') as f2:
            for i in total_step_history:
                f2.write(str(i) + ' ')
        print('Save the results...')
        return

    def _resetEValue(self):
        for value_dic in self.E.values():
            for action in range(self.env.action_space.n):
                value_dic[action] = 0.00


    def _get_(self, QorE, s, a):
        self._assert_state_in_QE(s, randomized=True)
        return QorE[s][a]

    def _set_(self, QorE, s, a, value):
        self._assert_state_in_QE(s, randomized=True)
        QorE[s][a] = value





