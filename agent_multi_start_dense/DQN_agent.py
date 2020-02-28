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
from keras.models import Model
import json, logging


def dqn_model(goal_input_shape, agent_input_shape):
    """
    Building a Neural Network Model
    """
    # --- Block concatenate([goal(3*3), agent(5*5)])
    #  dense128 --> dense32 --> dense4
    goal_input = Input(shape=goal_input_shape)
    agent_input = Input(shape=agent_input_shape)
    x = concatenate([goal_input, agent_input])
    x = Dense(128, activation='relu', name='dense_128')(x)
    x = Dense(32, activation='relu', name='dense_32')(x)
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
        goal_input_shape, agent_input_shape  = (9, ), (25, )
        self.q_model = dqn_model(goal_input_shape, agent_input_shape)
        self.q_model.compile(optimizer='adam', loss='mse')
        # - target Q Network
        self.target_model = dqn_model(goal_input_shape, agent_input_shape)
        # --- Q Network weights filename
        self.weights_file = model_file
        self.save_weights_file = 'dense_model.h5'

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
        alpha_state_batch, agent_state_batch, q_values_batch = [], [], []
        for state, action, reward, next_state, done in trans_pieces:
            # policy prediction for a given state and get Q_max
            q_values = self.q_model.predict(state)
            q_value = self._get_target_q_value(next_state, reward, q_double=True)
            # correction on the Q value for the action used
            q_values[0][action] = reward if done else q_value
            # collect batch state-q_value mapping
            alpha_state_batch.append(state[0][0])
            agent_state_batch.append(state[1][0])
            q_values_batch.append(q_values[0])
        # train the Q-network
        train_history = self.q_model.fit([np.array(alpha_state_batch), np.array(agent_state_batch)],
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
        self.epsilon = 1.0
        self.gamma = gamma
        # --- iteratively applying decay exploration and exploitation
        self.epsilon_min = min_epsilon
        self.epsilon_decay = self.epsilon_min / self.epsilon
        self.epsilon_decay = self.epsilon_decay ** (1. / float(max_episodes))
        # --- record the train process
        total_steps, step_in_episode, num_episode = 0, 0, 0
        steps_history, rewards_history, epsilon_history, step_in_episode_history = list(), list(), list(), list()
        loss_record = list()
        self.lose_the_way = 0
        while num_episode < max_episodes:
            # --- update exploration-exploitation probability
            self.update_epsilon()
            # update the epsilon
            step_in_episode, total_reward = 0, 0
            is_done, mean_loss = False, float('inf')
            goal_states, agent_states = self.env.reset()  # get the initial states
            # observation the new states
            goal_states = np.reshape(goal_states, [9, ])
            agent_states = np.reshape(agent_states, [25, ])
            goal_states = goal_states[np.newaxis, :]
            agent_s0 = agent_states[np.newaxis, :]
            while True:
                # action based on the alpha agent new states
                s0 = [goal_states, agent_s0]
                agent_a0 = self.perform_policy(s0, self.epsilon)
                agent_s1, agent_r1, is_done, agent_info = self.env.step(agent_a0)
                # update states
                agent_s1 = np.reshape(agent_s1, [25, ])
                agent_s1 = agent_s1[np.newaxis, :]
                # update GUI
                # self.env.render()
                step_in_episode += 1
                total_reward += agent_r1
                # Store experience in deque
                s1 = [goal_states, agent_s1]
                self.replay_buffer.append(np.array([s0, agent_a0, agent_r1, s1, is_done]))
                if is_done:
                    print('-' * 40, 'new iteration', '-' * 40)
                    logging.info('Agent arrives at the goal !')
                    self.lose_the_way = 0
                    break
                if step_in_episode > 1000:
                    # different maze size modify the param 1000
                    print('-'*30)
                    print('... Lose the way ...')
                    print('-'*30)
                    self.lose_the_way += 1
                    break
                agent_s0 = agent_s1
            if self.lose_the_way == 10:
                logging.info('Cannot find the way')
                input('stop...')
                return
            if len(self.replay_buffer) > batch_size:
                train_history = self._learn_from_memory(batch_size)
                mean_loss =  np.mean(train_history.history['loss']) / len(train_history.history['loss'])
                loss_record.append(mean_loss)

            # record the training results
            logging.info("episode: {:d}/{:d} | steps: {:d} | loss: {:.4f} | epsilon: {:.2f}"
                         .format(num_episode+1, max_episodes, step_in_episode, mean_loss, self.epsilon))
            logging.info('Episode reward: {:.2f}'.format(total_reward))
            total_steps += step_in_episode
            num_episode += 1
            steps_history.append(total_steps)
            step_in_episode_history.append(step_in_episode)
            rewards_history.append(total_reward)

            if len(step_in_episode_history) > 100 and np.mean(step_in_episode_history[-20:])<15:
                logging.info('Find the optimal path !')
                break

        print('Saving the model params...')
        # save Q Network params to a file
        self.q_model.save_weights(self.save_weights_file)

        # plot training rewards
        '''
          json.dumps()和json.loads()是json格式处理函数（可以这么理解，json是字符串）
          (0)json.dumps()函数是将一个Python数据类型列表进行json格式的编码（可以这么理解，json.dumps()函数是将字典转化为字符串）
          (1)json.loads()函数是将json格式数据转换为字典（可以这么理解，json.loads()函数是将字符串转化为字典）
        '''
        # steps_history
        c_list = json.dumps(steps_history)
        a = open(r"steps_history_5.txt", "w", encoding='UTF-8')
        a.write(c_list)
        a.close()
        # step_in_episode_history
        c_list = json.dumps(step_in_episode_history)
        a = open(r"step_in_episode_history_5.txt", "w", encoding='UTF-8')
        a.write(c_list)
        a.close()
        # loss
        c_list = json.dumps(loss_record)
        a = open(r"loss_5.txt", "w", encoding='UTF-8')
        a.write(c_list)
        a.close()

        plt.figure('Step')
        plt.plot(steps_history, step_in_episode_history)
        plt.xlabel('Total steps')
        # plot loss function
        plt.figure('Loss')
        plt.plot(range(1, len(loss_record)+1),loss_record)
        plt.xlabel('Episode')
        plt.ylabel('Loss value')
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
