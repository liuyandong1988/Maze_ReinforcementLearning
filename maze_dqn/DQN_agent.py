#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Yandong
# Time  :2019/12/7 20:42
from gym import Env, spaces
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from keras.optimizers import Adam
from keras.layers import Dense, Input, PReLU
from keras.models import Model, Sequential
from keras.callbacks import TensorBoard
from keras.models import load_model



def dqn_model(input_dim, hidden_dim, out_dim):
    """
    Building a Neural Network Model
    """
    model = Sequential()
    model.add(Dense(input_dim, input_shape=(input_dim,), activation='relu'))
    model.add(Dense(hidden_dim, activation='relu'))
    model.add(Dense(out_dim))
    model.summary()
    return model

class QAgent(object):

    def __init__(self, env: Env = None, memory_capacity=2000, hidden_dim=100, model_file=False):
        if env is None:
            raise Exception("agent should have an environment")
        if isinstance(env.observation_space, spaces.Discrete):
            self.input_dim = env.observation_space.n
        elif isinstance(env.observation_space, spaces.Box):
            self.input_dim = env.observation_space.shape[0]
        if isinstance(env.action_space, spaces.Discrete):
            self.output_dim = env.action_space.n
        elif isinstance(env.action_space, spaces.Box):
            self.output_dim = env.action_space.shape[0]

        self.env = env
        # network update
        self.replay_counter = 1
        self.replay_buffer = list()
        self.replay_buffer_capacity = memory_capacity
        if model_file:
            self.q_model = load_model(model_file)
        else:
            # the double DQN, q_model and q_target_model
            self.q_model = dqn_model(self.input_dim, hidden_dim, self.output_dim)
            self.q_model.compile(optimizer='adam', loss='mse')
        # target Q Network
        self.target_q_model = dqn_model(self.input_dim, hidden_dim, self.output_dim)
        # copy Q Network params to target Q Network
        self._update_weights()
        self.tbCallBack = TensorBoard(log_dir='./logs',  # log 目录
                                      # histogram_freq=0,  # 按照何等频率（epoch）来计算直方图，0为不计算
                                      # batch_size=32,     # 用多大量的数据计算直方图
                                      write_graph=True,  # 是否存储网络结构图
                                      write_grads=True,  # 是否可视化梯度直方图
                                      write_images=True,  # 是否可视化参数
                                      embeddings_freq=0,
                                      embeddings_layer_names=None,
                                      embeddings_metadata=None)


    # copy trained Q Network params to target Q Network
    def _update_weights(self):
        self.target_q_model.set_weights(self.q_model.get_weights())

    # compute Q_max
    # use of target Q Network solves the non-stationarity problem
    def _get_target_q_value(self, next_state, reward, q_double=False):
        # max Q value among next state's actions
        if q_double:
            # DDQN
            # current Q Network selects the action
            # a'_max = argmax_a' Q(s', a')
            action = np.argmax(self.q_model.predict(next_state)[0])
            # target Q Network evaluates the action
            # Q_max = Q_target(s', a'_max)
            q_value = self.target_q_model.predict(next_state)[0][action]
        else:
            # DQN chooses the max Q value among next actions
            # selection and evaluation of action is on the target Q Network
            # Q_max = max_a' Q_target(s', a')
            q_value = np.amax(self.target_q_model.predict(next_state)[0])
        # Q_max = reward + gamma * Q_max
        q_value *= self.gamma
        q_value += reward
        return q_value

    def _learn_from_memory(self, batch_size):
        # Sample experience
        trans_pieces = random.sample(self.replay_buffer, batch_size)  # the transition <s0, a0, r1, is_done, s1>
        state_batch, q_values_batch = [], []
        for state, action, reward, next_state, done in trans_pieces:
            # policy prediction for a given state
            q_values = self.q_model.predict(state)
            # get Q_max
            q_value = self._get_target_q_value(next_state, reward)
            # correction on the Q value for the action used
            q_values[0][action] = reward if done else q_value
            # collect batch state-q_value mapping
            state_batch.append(state[0])
            q_values_batch.append(q_values[0])
        # train the Q-network
        self.q_model.fit(np.array(state_batch),
                         np.array(q_values_batch),
                         batch_size=32, epochs=16, verbose=0)
        # self.q_model.fit(np.array(state_batch),
        #                  np.array(q_values_batch),
        #                  batch_size=32, epochs=16, verbose=0,
        #                  callbacks=[self.tbCallBack])
        loss = self.q_model.evaluate(np.array(state_batch), np.array(q_values_batch), verbose=0)
        # the target_net update
        self._update_weights()
        # if self.replay_counter % 10 == 0:
        #     self._update_weights()
        # self.replay_counter += 1
        return loss

    def act(self, a0, s0):
        s1, r1, is_done, info = self.env.step(a0)
        s1 = np.reshape(s1, [1, self.input_dim])
        # put the <s0, a0, r1, is_done, s1> in the memory
        # Store experience in deque
        self.replay_buffer.append(np.array([s0, a0, r1, s1, is_done]))
        if len(self.replay_buffer) > self.replay_buffer_capacity:
            self.replay_buffer.pop(0)
        return s1, r1, is_done, info

    def learning(self, max_episodes=1000, batch_size=32, gamma=0.99, min_epsilon=0.1):
        """
        epsilon-greed find the action and experience replay
        :return:
        """
        # initially 90% exploration, 10% exploitation
        self.epsilon = 1.0
        self.gamma = gamma
        # iteratively applying decay til 10% exploration/90% exploitation
        self.epsilon_min = min_epsilon
        self.epsilon_decay = self.epsilon_min / self.epsilon
        self.epsilon_decay = self.epsilon_decay ** (1. / float(max_episodes))
        total_steps, step_in_episode, num_episode = 0, 0, 0
        steps_history, rewards_history, epsilon_history = list(), list(), list()
        # self.min_reward = -10 * self.env.maze_size
        while num_episode < max_episodes:
            # update exploration-exploitation probability
            self.update_epsilon()
            # self.update_epsilon(num_episode)  # 2
            epsilon_history.append(self.epsilon)
            # update the epsilon
            step_in_episode, total_reward = 0, 0
            loss, mean_loss = 0, 0
            is_done = False
            env_state = self.env.reset()
            s0 = np.reshape(env_state, [1, self.input_dim])
            while not is_done:
                a0 = self.perform_policy(s0, self.epsilon)
                self.env.render()
                s1, r1, is_done, info = self.act(a0, s0)
                total_reward += r1
                # if total_reward < self.min_reward:
                #     is_done = True
                step_in_episode += 1
                s0 = s1
            # call experience relay
            if len(self.replay_buffer) > batch_size:
                loss += self._learn_from_memory(batch_size)
            mean_loss = loss / step_in_episode
            print("episode: {:03d}/{:d} time_step:{:d} epsilon:{:3.2f}, loss:{:.5f}"
                  .format(num_episode+1, max_episodes, step_in_episode, self.epsilon, mean_loss))
            print('Episode reward: {:.2f}'.format(total_reward))
            if (num_episode+1) % 50 == 0:
                # save Q Network params to a file
                # Q Network weights filename
                self.weights_file = 'dqn_maze_' + str(num_episode+1) + '.h5'
                self.q_model.save('./model3/' + self.weights_file)
                print("Saved Model")
            total_steps += step_in_episode
            num_episode += 1
            steps_history.append(total_steps)   # the total step
            rewards_history.append(step_in_episode)  #  the same as reward

        # store the weight and score result
        with open('time_in_episode_history_dqn.txt', 'w') as f1:
            for i in rewards_history:
                f1.write(str(i) + ' ')
        with open('total_step_history_dqn.txt', 'w') as f2:
            for i in steps_history:
                f2.write(str(i) + ' ')
        with open('epsilon.txt', 'w') as f3:
            for i in epsilon_history:
                f3.write(str(i) + ' ')
        print('Save the results...')
        # plot training rewards
        plt.plot(steps_history, rewards_history)
        plt.xlabel('Steps')
        plt.ylabel('Steps in one episode')
        plt.show()
        return

    # decrease the exploration, increase exploitation
    def update_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    # def update_epsilon(self, episode):
    #     self.epsilon = 1 / (1+episode)


    def perform_policy(self, s, epsilon=None):
        """
        New action based on the Q_update net
        """
        Q_s = self.q_model.predict(s)[0]
        if epsilon is not None and random.random() < epsilon:
            action = self.env.action_space.sample()
            return action
        else:
            return int(np.argmax(Q_s))
