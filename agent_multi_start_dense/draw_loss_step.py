#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Yandong
# Time: 2020/2/21 19:48
# draw the loss and step
import json
import matplotlib.pyplot as plt


# file = open('loss_1.txt', 'r')
# js = file.read()
# loss_1 = json.loads(js)
#
# file = open('loss_2.txt', 'r')
# js = file.read()
# loss_2 = json.loads(js)
#
# file = open('loss_3.txt', 'r')
# js = file.read()
# loss_3 = json.loads(js)
#
# file = open('loss_4.txt', 'r')
# js = file.read()
# loss_4 = json.loads(js)

file = open('loss_5.txt', 'r')
js = file.read()
loss_5 = json.loads(js)
# draw the step
# file = open('step_in_episode_history_1.txt', 'r')
# js = file.read()
# step_1 = json.loads(js)
#
# file = open('step_in_episode_history_2.txt', 'r')
# js = file.read()
# step_2 = json.loads(js)
#
# file = open('step_in_episode_history_3.txt', 'r')
# js = file.read()
# step_3 = json.loads(js)
#
# file = open('step_in_episode_history_4.txt', 'r')
# js = file.read()
# step_4 = json.loads(js)

file = open('step_in_episode_history_5.txt', 'r')
js = file.read()
step_5 = json.loads(js)

plt.figure('loss')
# plt.plot(loss_1, label='ex1_diver')
# plt.plot(loss_2, label='ex2_diver')
# plt.plot(loss_3, label='ex3_diver')
# plt.plot(loss_4, label='ex4_diver')
plt.plot(loss_5, label='ex5_convergence')
plt.ylabel('Loss')
plt.xlabel('Iteration: 500')
plt.legend(loc='lower right')
plt.show()

plt.figure('step')
# plt.plot(step_1, label='ex1_diver')
# plt.plot(step_2, label='ex2_diver')
# plt.plot(step_3, label='ex3_diver')
# plt.plot(step_4, label='ex4_diver')
plt.plot(step_5, label='ex5_convergence')
plt.ylabel('Step in each episode')
plt.xlabel('Iteration: 500')
plt.legend(loc='lower right')
plt.show()
