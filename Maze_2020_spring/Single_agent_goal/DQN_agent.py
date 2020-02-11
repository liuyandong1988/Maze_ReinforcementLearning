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
import gc

def dqn_model(input_dim, hidden_dim, out_dim):
    """
    Building a Neural Network Model
    """
    model = Sequential()
    model.add(Dense(hidden_dim, input_shape=(input_dim,)))
    model.add(PReLU())
    model.add(Dense(int(hidden_dim/2)))
    model.add(PReLU())
    model.add(Dense(int(hidden_dim/4)))
    model.add(PReLU())
    model.add(Dense(out_dim))
    model.summary()
    return model

class QAgent(object):

    def __init__(self, env: Env = None, memory_capacity=2000, hidden_dim=100, model_file=None):
        if env is None:
            raise Exception("agent should have an environment")
        self.input_dim = env.observation_space.n
        self.output_dim = env.action_space.n
        self.env = env
        # replay experiment parameters
        self.replay_counter = 1
        self.replay_buffer = list()
        self.replay_buffer_capacity = memory_capacity
        # Q Network weights filename
        self.load_weights_file = model_file
        self.save_weights_file = 'single_agent_5_goals.h5'
        # the double DQN, q_model and q_target_model
        self.q_model = dqn_model(self.input_dim, hidden_dim, self.output_dim)
        self.q_model.compile(optimizer='adam', loss='mse')
        # target Q Network
        self.target_q_model = dqn_model(self.input_dim, hidden_dim, self.output_dim)
        if self.load_weights_file:
            self.q_model.load_weights(self.load_weights_file)
        else:
            pass
        # copy Q Network params to target Q Network
        self._update_weights()

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
        self.epsilon_decay = self.epsilon_decay ** (1. / float(max_episodes*2))
        total_steps, step_in_episode, num_episode = 0, 0, 0
        steps_history, rewards_history, epsilon_history, step_in_episode_history = list(), list(), list(), list()
        # env state, action, next state
        env_state = list()
        # self.min_reward = -10 * self.env.maze_size
        while num_episode < max_episodes:
            # update exploration-exploitation probability
            self.update_epsilon()
            # self.update_epsilon(num_episode)  # 2 method
            # epsilon_history.append(self.epsilon)
            # update the epsilon
            step_in_episode, total_reward = 0, 0
            loss, mean_loss = 0, 0
            is_done = False
            env_state.clear()
            self.env.reset(env_state)  # get the env observation states
            print('goal', self.env.goal_pos)
            # print(id(env_state))
            s0 = np.reshape(env_state, [1, self.input_dim])
            while not is_done:
                a0 = self.perform_policy(s0, self.epsilon)
                # a0 = int(np.argmax(self.q_model.predict(s0)[0]))
                self.env.render()
                s1, r1, is_done, info = self.act(a0, s0)
                total_reward += r1
                # if total_reward < self.min_reward:
                #     is_done = True
                step_in_episode += 1
                s0 = s1
                # gc.collect()
            # call experience relay
            if len(self.replay_buffer) > batch_size:
                loss += self._learn_from_memory(batch_size)
            mean_loss = loss / step_in_episode
            print("episode: {:03d}/{:d} time_step:{:d} epsilon:{:3.2f}, loss:{:.5f}"
                  .format(num_episode+1, max_episodes, step_in_episode, self.epsilon, mean_loss))
            print('Episode reward: {:.2f}'.format(total_reward))
            total_steps += step_in_episode
            num_episode += 1
            steps_history.append(total_steps)
            step_in_episode_history.append(step_in_episode)
            rewards_history.append(total_reward)

            # finishing condition...
            # if len(rewards_history) > 20 and np.mean(rewards_history[-20:]) > 36:
            #     print('Saving the model params...')
            #     # save Q Network params to a file
            #     self.q_model.save_weights(self.save_weights_file)
            #     print('Finish training !')
            #     break
        print('Saving the model params...')
        # save Q Network params to a file
        self.q_model.save_weights(self.save_weights_file)
        # plot training rewards
        plt.plot(steps_history, step_in_episode_history)
        plt.xlabel('steps')
        plt.ylabel('running avg steps')
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
