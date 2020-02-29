#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Yandong
# Time  :2019/12/7 20:42
from gym import Env, spaces
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from keras.layers import Dense, Input, concatenate
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten
from keras.models import Model
import json


def dqn_model(goal_input_shape, agent_input_shape):
    """
    Building a Neural Network Model
    """
    # --- Block
    # goal_input (batch, 3, 3, 1) --> batch*160
    goal_input = Input(shape=goal_input_shape)
    goal_x = Conv2D(32, (3, 3), activation='relu', padding='same', name='goal_conv')(goal_input)
    # goal_x = MaxPooling2D(pool_size=(2, 2))(goal_x)
    # goal_x = Dropout(0.1)(goal_x)
    goal_x = Flatten()(goal_x)
    # agent_input (batch, 5, 5, 3) --> batch*448
    agent_input = Input(shape=agent_input_shape)
    agent_x = Conv2D(32, (3, 3), activation='relu', padding='same', name='agent_conv')(agent_input)
    # agent_x = MaxPooling2D(pool_size=(2, 2))(agent_x)
    # agent_x = Dropout(0.1)(agent_x)
    agent_x = Flatten()(agent_x)
    # concatenate
    # [batch*160, batch*448] --> batch*608
    x = concatenate([goal_x, agent_x])
    #  dense: batch*608 --> batch*128 --> batch*4
    x = Dense(128, activation='relu', name='dense_128')(x)
    output_data = Dense(4, activation='relu', name='dense_4')(x)
    model = Model(inputs=[goal_input, agent_input], outputs=output_data)
    model.summary()
    return model


class QAgent(object):

    def __init__(self, env: Env = None, memory_capacity=2000, model_file=None):
        if env is None:
            raise Exception("agent should have an environment")
        self.env = env
        # --- replay experiment parameters
        self.replay_counter = 1
        self.replay_buffer = deque(maxlen=memory_capacity)
        # --- the double DQN, model and target_model
        goal_input_shape, agent_input_shape = (3, 3, 1), (5, 5, 3)
        self.q_model = dqn_model(goal_input_shape, agent_input_shape)
        self.q_model.compile(optimizer='adam', loss='mse')
        # - target Q Network
        self.target_model = dqn_model(goal_input_shape, agent_input_shape)
        # --- Q Network weights filename
        self.weights_file = model_file
        self.loss_time = 0
        self.save_weights_file = 'convolution_model.h5'

        if self.weights_file:
            print('Load the model parameter ... ')
            self.q_model.load_weights(self.weights_file)
        else:
            pass
        # copy Q Network params to target Q Network
        self._update_weights()

    def _update_weights(self):
        """
        copy trained Q Network params to target Q Network
        """
        self.target_model.set_weights(self.q_model.get_weights())

    def _get_target_q_value(self, next_state, reward, q_double=False):
        # max Q value among next state's actions
        if q_double:
            action = np.argmax(self.q_model.predict(next_state)[0])
            q_value = self.target_model.predict(next_state)[0][action]
        else:
            # DQN chooses the max Q value among next actions
            # selection and evaluation of action is on the target Q Network
            # Q_max = max_a' Q_target(s', a')
            q_value = np.amax(self.target_model.predict(next_state)[0])
        # Q_max = reward + gamma * Q_max
        q_value *= self.gamma
        q_value += reward
        return q_value

    def _learn_from_memory(self, batch_size):
        # Sample experience
        trans_pieces = random.sample(self.replay_buffer, batch_size)
        goal_state_batch, agent_state_batch, q_values_batch = [], [], []
        for state, action, reward, next_state, done in trans_pieces:
            # policy prediction for a given state and get Q_max
            q_values = self.q_model.predict(state)
            q_value = self._get_target_q_value(next_state, reward, q_double=True)
            # correction on the Q value for the action used
            q_values[0][action] = reward if done else q_value
            # collect batch state-q_value mapping
            goal_state_batch.append(state[0][0])
            agent_state_batch.append(state[1][0])
            q_values_batch.append(q_values[0])
        # train the Q-network
        train_history = self.q_model.fit([np.array(goal_state_batch), np.array(agent_state_batch)],
                                          np.array(q_values_batch),
                                          batch_size=64, epochs=16, verbose=0)
        # the target_net update
        if self.replay_counter % 5 == 0:
            print('* Update the target model !')
            self._update_weights()
        self.replay_counter += 1
        return train_history

    def learning(self, max_episodes=1000, batch_size=32, gamma=0.99, min_epsilon=0.01):
        """
        epsilon-greed find the action and experience replay
        :return:
        """
        lose_the_way = False
        self.epsilon = 1.0
        self.gamma = gamma
        # --- iteratively applying decay exploration and exploitation
        self.epsilon_min = min_epsilon
        self.epsilon_decay = self.epsilon_min / self.epsilon
        self.epsilon_decay = self.epsilon_decay ** (1. / float(max_episodes))
        # self.epsilon_decay = 1
        # --- record the train process
        total_steps, step_in_episode, num_episode = 0, 0, 0
        steps_history, rewards_history, epsilon_history, step_in_episode_history = list(), list(), list(), list()
        # --- initial agent states
        agent_init_states = np.ones((5, 5, 1))
        agent_3_states = deque(maxlen=3)
        while num_episode < max_episodes:
            # 3 continue state
            for _ in range(3):
                agent_3_states.append(agent_init_states)
            # --- update exploration-exploitation probability
            self.update_epsilon()
            step_in_episode, total_reward = 0, 0
            goal_states, agent_states = self.env.reset()  # get the initial states
            # --- observation the new states
            goal_states = np.reshape(goal_states, [3, 3, 1])
            goal_states = goal_states[np.newaxis, :]
            self.goal_states = goal_states
            agent_states = np.reshape(agent_states, [5, 5, 1])
            agent_3_states.append(agent_states)
            # reshape the dim [3, 5, 5, 1] --> [1, 5, 5, 3]
            agent_s0 = np.concatenate((agent_3_states[0], agent_3_states[1], agent_3_states[2]), axis=2)
            agent_s0 = agent_s0[np.newaxis, :]
            while True:
                # action based on the alpha agent new states
                s0 = [goal_states, agent_s0]
                agent_a0 = self.perform_policy(s0, self.epsilon)
                agent_s1, agent_r1, is_done, agent_info = self.env.step(agent_a0)
                # update agent states
                agent_states = np.reshape(agent_s1, [5, 5, 1])
                agent_3_states.append(agent_states)
                # reshape the dim [3, 5, 5, 1] --> [1, 5, 5, 3]
                agent_s1 = np.concatenate((agent_3_states[0], agent_3_states[1], agent_3_states[2]), axis=2)
                agent_s1 = agent_s1[np.newaxis, :]
                # update GUI
                # self.env.render()
                step_in_episode += 1
                total_reward += agent_r1
                # Store experience in deque
                s1 = [goal_states, agent_s1]
                self.replay_buffer.append(np.array([s0, agent_a0, agent_r1, s1, is_done]))
                if is_done:
                    lose_the_way = False
                    print('-' * 30, 'new iteration', '-' * 30)
                    print('agent meets Alpha ...')
                    break
                if step_in_episode > 1000:
                    # different maze size modify the param 1000
                    print('-'*30)
                    print('... Lose the way ...')
                    print('-'*30)
                    lose_the_way = True
                    self.loss_time += 1
                    break
                agent_s0 = agent_s1
            if self.loss_time == 10:
                print('Cannot find the way.')
                input('Stop')
            if len(self.replay_buffer) > batch_size:
                train_history = self._learn_from_memory(batch_size)
                print('Loss: %.05f' % np.mean(train_history.history['loss']))
                # print('333:', len(train_history.history['loss']))
            print("Episode: {:03d}/{:d} time_step:{:d} epsilon:{:3.2f}"
                  .format(num_episode+1, max_episodes, step_in_episode, self.epsilon))
            print('Episode reward: {:.2f}'.format(total_reward))
            total_steps += step_in_episode
            num_episode += 1
            steps_history.append(total_steps)
            step_in_episode_history.append(step_in_episode)
            rewards_history.append(total_reward)

            if len(step_in_episode_history) > 100 and np.mean(step_in_episode_history[-20:])<12:
                print('Find the optimal path !')
                break

        print('Saving the model params...')
        # save Q Network params to a file
        # self.alpha_q_model.save_weights(self.alpha_save_weights_file)
        self.q_model.save_weights(self.save_weights_file)
        # save the train step in file

        '''
          json.dumps()和json.loads()是json格式处理函数（可以这么理解，json是字符串）
          (1)json.dumps()函数是将一个Python数据类型列表进行json格式的编码（可以这么理解，json.dumps()函数是将字典转化为字符串）
          (2)json.loads()函数是将json格式数据转换为字典（可以这么理解，json.loads()函数是将字符串转化为字典）
        '''
        # steps_history
        c_list = json.dumps(steps_history)
        a = open(r"steps_history_1.txt", "w", encoding='UTF-8')
        a.write(c_list)
        a.close()
        # step_in_episode_history
        c_list = json.dumps(step_in_episode_history)
        a = open(r"step_in_episode_history_1.txt", "w", encoding='UTF-8')
        a.write(c_list)
        a.close()





        # plot training rewards
        plt.plot(steps_history, step_in_episode_history)
        plt.xlabel('Total steps')
        plt.ylabel('Steps in each episode')
        plt.show()
        return

    def update_epsilon(self):
        """
        decrease the exploration, increase exploitation
        """
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

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
