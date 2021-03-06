#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Yandong
# Time  :2019/12/3 7:05

from gym import Env
from random import *
import datetime
from gym import spaces
from gym.utils import seeding
import numpy as np


class Grid(object):
    def __init__(self, x: int = None,
                 y: int = None,
                 type: int = 0,
                 reward: int = 0.0,
                 value: float = 0.0):  # value, for possible future usage
        self.x = x  # coordinate x
        self.y = y
        self.type = type  # Type (1:empty；0:obstacle or boundary)
        self.reward = reward  # instant reward for an agent entering this grid cell
        self.value = value  # the value of this grid cell, for future usage
        self.name = None  # name of this grid.
        self._update_name()

    def _update_name(self):
        self.name = "X{0}-Y{1}".format(self.x, self.y)

    def __str__(self):
        return "name:{4}, x:{0}, y:{1}, type:{2}, value{3}".format(self.x,
                                                                   self.y,
                                                                   self.type,
                                                                   self.reward,
                                                                   self.value,
                                                                   self.name
                                                                   )


class GridMatrix(object):
    '''
    格子矩阵，通过不同的设置，模拟不同的格子世界环境
    '''

    def __init__(self, n_width: int,  # defines the number of cells horizontally
                 n_height: int,  # vertically
                 default_type: int = 0,  # default cell type
                 default_reward: float = 0.0,  # default instant reward
                 default_value: float = 0.0  # default value
                 ):
        self.grids = None
        self.n_height = n_height
        self.n_width = n_width
        self.len = n_width * n_height
        self.default_reward = default_reward
        self.default_value = default_value
        self.default_type = default_type
        self.reset()

    def reset(self):
        self.grids = []
        for x in range(self.n_height):
            for y in range(self.n_width):
                self.grids.append(Grid(x,
                                       y,
                                       self.default_type,
                                       self.default_reward,
                                       self.default_value))

    def get_grid(self, x, y=None):
        '''get a grid information
        args: represented by x,y or just a tuple type of x
        return: grid object
        '''
        xx, yy = None, None
        if isinstance(x, int):
            xx, yy = x, y
        elif isinstance(x, tuple):
            xx, yy = x[0], x[1]
        assert (xx >= 0 and yy >= 0 and xx < self.n_width and yy <
                self.n_height), "coordinates should be in reasonable range"
        index = yy * self.n_width + xx
        return self.grids[index]

    def set_reward(self, x, y, reward):
        grid = self.get_grid(x, y)
        if grid is not None:
            grid.reward = reward
        else:
            raise ("grid doesn't exist")

    def set_value(self, x, y, value):
        grid = self.get_grid(x, y)
        if grid is not None:
            grid.value = value
        else:
            raise ("grid doesn't exist")

    def set_type(self, x, y, type):
        grid = self.get_grid(x, y)
        if grid is not None:
            grid.type = type
        else:
            raise ("grid doesn't exist")

    def get_reward(self, x, y):
        grid = self.get_grid(x, y)
        if grid is None:
            return None
        return grid.reward

    def get_value(self, x, y):
        grid = self.get_grid(x, y)
        if grid is None:
            return None
        return grid.value

    def get_type(self, x, y):
        grid = self.get_grid(x, y)
        if grid is None:
            return None
        return grid.type


class GridWorldEnv(Env):
    """
   格子世界环境，可以模拟各种不同的格子世界
    """
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self,
                 n_width: int = 10,
                 n_height: int = 10,
                 u_size=40,
                 default_reward: float = 0,
                 default_type=1):
        self.u_size = u_size  # size for each cell (pixels)
        # width of the env calculated by number of cells.
        self.n_width = n_width
        self.n_height = n_height  # height...
        self.maze_size = n_width * n_height
        self.width = u_size * n_width  # scenario width (pixels)
        self.height = u_size * n_height  # height
        self.default_reward = default_reward
        self.default_type = default_type
        self.obstacles = list()
        self.grids = GridMatrix(n_width=self.n_width,
                                n_height=self.n_height,
                                default_reward=self.default_reward,
                                default_type=self.default_type,
                                default_value=0.0)

        self.action = None  # for rendering
        # 0,1,2,3 represent left, right, up, down, -,moves.
        self.action_space = spaces.Discrete(4)
        # 坐标原点为左下角，这个pyglet是一致的, left-bottom corner is the position of (0,0)
        # 通过设置起始点、终止点以及特殊奖励和类型的格子可以构建各种不同类型的格子世界环境
        # 比如：随机行走、汽车租赁、悬崖行走等David Silver公开课中的示例
        self.starts = []  # start cell position
        self.end = None
        self.types = []  # special grid [(3,2,1)] position (3,2) value:1
        # special type of cells, (x,y,z) represents in position(x,y) the cell
        # type is z
        self.rewards = []  # 特殊奖励的格子在此设置，终止格子奖励0, special reward for a cell
        self.visited = set()  # visit nodes set
        self.viewer = None  # 图形接口对象
        # self.seed()  # 产生一个随机子
        self.choose_start = 0

    def seed(self, seed=None):
        """
        repeat
        """
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        """
        moving based on action
        """
        assert self.action_space.contains(action), \
            "%r (%s) invalid" % (action, type(action))
        # --- action
        self.action = action  # action for rendering
        old_x, old_y = self.agent_states
        new_x, new_y = old_x, old_y
        if action == 0:
            new_x -= 1  # left
        elif action == 1:
            new_x += 1  # right
        elif action == 2:
            new_y += 1  # up
        elif action == 3:
            new_y -= 1  # down
        # --- get the local observation states and reward
        reward = 0
        distance = (np.abs(self.goal_position[0] - new_x) +
                    np.abs(self.goal_position[1] - new_y))
        if distance == 0:
            reward_distance = self.n_width + 1
         # reward_distance = self.start_end_distance/2  + 1
         #    reward_distance = -distance
        else:
            reward_distance = self.n_width / distance
            # reward_distance = (self.start_end_distance ) /(2 * distance)
            # reward_distance = -distance
        reward += reward_distance
        # --- wall effect, obstacles or boundary.
        collision_mark = False
        # boundary effect
        if new_x < 0 or new_x >= self.n_width or new_y < 0 or new_y >= self.n_height:
            new_x, new_y = old_x, old_y
            collision_mark = True
        elif self.grids.get_type(new_x, new_y) == 0:
            new_x, new_y = old_x, old_y
            collision_mark = True
        done = self._is_end_state(new_x, new_y)
        # --- get the local observation states
        self.agent_states = (new_x, new_y)
        agent_state_cell = self._xy_to_state(self.agent_states)
        states = self.agent_get_local_state()
        # --- get the reward
        if agent_state_cell in self.visited:
            if collision_mark:
                reward -= 5   # visit the boundary or obstacles
            else:
                reward -= 2 # repeat the visited ceil
        if done:
            reward += 100     # goal
        reward -= 1  # pass a ceil
        # --- add the visited city
        if agent_state_cell not in self.visited:
            self.visited.add(agent_state_cell)  # the past cell
        # all information
        info = {"x": new_x, "y": new_y, "grids": self.grids}
        return states, reward, done, info

    # set status into an one-axis coordinate value
    def _state_to_xy(self, s):
        x = s % self.n_width
        y = int((s - x) / self.n_width)
        return x, y

    def _xy_to_state(self, x, y=None):
        if isinstance(x, int):
            assert (isinstance(y, int)), "incomplete Position info"
            return x + self.n_width * y
        elif isinstance(x, tuple):
            return x[0] + self.n_width * x[1]
        return -1  # 未知状态, unknow status

    def get_map_state(self):
        # observation states
        maze_env = np.ones((self.n_width, self.n_height))
        for g in self.grids.grids:
            maze_env[g.x][g.y] = g.type
        # self.maze_env_states = maze_env[::-1]
        self.maze_env_states = maze_env.T

    def goal_local_state(self):
        """
        get the 9 local states from agent position and maze
        """
        x, y = self.goal_position
        local_state_pos = [(x - 1, y - 1), (x - 1, y), (x - 1, y + 1),
                           (x, y - 1), (x, y), (x, y + 1),
                           (x + 1, y - 1), (x + 1, y), (x + 1, y + 1)]
        local_state = list()
        for pos in local_state_pos:
            if pos[0] == -1 or pos[1] == -1:
                local_state.append(0)
            else:
                try:
                    self.maze_env_states[pos[0], pos[1]]
                except IndexError:
                    local_state.append(0)
                else:
                    local_state.append(self.maze_env_states[pos[0], pos[1]])
        # the distance from agent to goal
        return local_state

    def agent_get_local_state(self):
        """
        get the 25 local states from agent position and maze
        """
        x, y = self.agent_states
        local_state_pos = [(x -
                            2, y -
                            2), (x -
                                 2, y -
                                 1), (x -
                                      2, y), (x -
                                              2, y +
                                              1), (x -
                                                   2, y +
                                                   2), (x -
                                                        1, y -
                                                        2), (x -
                                                             1, y -
                                                             1), (x -
                                                                  1, y), (x -
                                                                          1, y +
                                                                          1), (x -
                                                                               1, y +
                                                                               2), (x, y -
                                                                                    2), (x, y -
                                                                                         1), (x, y), (x, y +
                                                                                                      1), (x, y +
                                                                                                           2), (x +
                                                                                                                1, y -
                                                                                                                2), (x +
                                                                                                                     1, y -
                                                                                                                     1), (x +
                                                                                                                          1, y), (x +
                                                                                                                                  1, y +
                                                                                                                                  1), (x +
                                                                                                                                       1, y +
                                                                                                                                       2), (x +
                                                                                                                                            2, y -
                                                                                                                                            2), (x +
                                                                                                                                                 2, y -
                                                                                                                                                 1), (x +
                                                                                                                                                      2, y), (x +
                                                                                                                                                              2, y +
                                                                                                                                                              1), (x +
                                                                                                                                                                   2, y +
                                                                                                                                                                   2)]
        local_state = list()
        for pos in local_state_pos:
            if pos[0] < 0 or pos[1] < 0:
                local_state.append(0)
            else:
                try:
                    self.maze_env_states[pos[0], pos[1]]
                except IndexError:
                    local_state.append(0)
                else:
                    local_state.append(self.maze_env_states[pos[0], pos[1]])
        # the distance from agent to goal
        return local_state

    def reset(self, start=None, test=False):
        """
        choose the start and reset parameters
        """
        if not test:
            start_num = len(self.starts)
            self.choose_start += 1
            self.agent_states = self.starts[self.choose_start % start_num]
            # end
            self.goal_position = self.end
            self.start_end_distance = (
                np.abs(
                    self.agent_states[0] -
                    self.end[0]) +
                np.abs(
                    self.agent_states[1] -
                    self.end[1]))
        else:
            self.agent_states = start
            self.goal_position = self.end
        self.visited.clear()
        # the start position
        self.visited.add(self._xy_to_state(self.agent_states))
        goal_states = self.goal_local_state()
        agent_states = self.agent_get_local_state()
        return goal_states, agent_states

    def refresh_setting(self):
        """
        grid type property
        """
        for x, y, r in self.rewards:
            self.grids.set_reward(x, y, r)  # the goal
        for x, y, t in self.types:
            self.grids.set_type(x, y, t)  # the obstacle
        self.get_map_state()
        # initialization
        self.reset()

    def _is_end_state(self, x, y):
        """
        Arrive the goal
        """
        if (x, y) == self.end:
            return True
        else:
            return False

    def render(self, mode='human', close=False):
        """
        Graphic UI by openAI gym
        """
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return
        zero = (0, 0)
        u_size = self.u_size
        m = 2  # gaps between two cells

        # 如果还没有设定屏幕对象，则初始化整个屏幕具备的元素。
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(self.width, self.height)

            # 在Viewer里绘制一个几何图像的步骤如下：
            # the following steps just tells how to render an shape in the environment.
            # 1. 建立该对象需要的数据本身
            # 2. 使用rendering提供的方法返回一个geom对象
            # 3. 对geom对象进行一些对象颜色、线宽、线型、变换属性的设置（有些对象提供一些个
            #    性化的方法来设置属性，具体请参考继承自这些Geom的对象），这其中有一个重要的
            #    属性就是变换属性，
            #    该属性负责对对象在屏幕中的位置、渲染、缩放进行渲染。如果某对象
            #    在呈现时可能发生上述变化，则应建立关于该对象的变换属性。该属性是一个
            #    Transform对象，而一个Transform对象，包括translate、rotate和scale
            #    三个属性，每个属性都由以np.array对象描述的矩阵决定。
            # 4. 将新建立的geom对象添加至viewer的绘制对象列表里，如果在屏幕上只出现一次，
            #    将其加入到add_onegeom(）列表中，如果需要多次渲染，则将其加入add_geom()
            # 5. 在渲染整个viewer之前，对有需要的geom的参数进行修改，修改主要基于该对象
            #    的Transform对象
            # 6. 调用Viewer的render()方法进行绘制
            ''' 绘制水平竖直格子线，由于设置了格子之间的间隙，可不用此段代码
            for i in range(self.n_width+1):
                line = rendering.Line(start = (i*u_size, 0),
                                      end =(i*u_size, u_size*self.n_height))
                line.set_color(0.5,0,0)
                self.viewer.add_geom(line)
            for i in range(self.n_height):
                line = rendering.Line(start = (0, i*u_size),
                                      end = (u_size*self.n_width, i*u_size))
                line.set_color(0,0,1)
                self.viewer.add_geom(line)
            '''

            # 绘制格子, draw cells
            for x in range(self.n_width):
                for y in range(self.n_height):
                    v = [(x * u_size + m, y * u_size + m),
                         ((x + 1) * u_size - m, y * u_size + m),
                         ((x + 1) * u_size - m, (y + 1) * u_size - m),
                         (x * u_size + m, (y + 1) * u_size - m)]

                    rect = rendering.FilledPolygon(v)
                    r = self.grids.get_reward(x, y) / 10
                    if r < 0:
                        rect.set_color(0.9 - r, 0.9 + r, 0.9 + r)
                    elif r > 0:
                        rect.set_color(0.3, 0.5 + r, 0.3)
                    else:
                        rect.set_color(0.9, 0.9, 0.9)
                    self.viewer.add_geom(rect)
                    # 绘制边框, draw frameworks
                    v_outline = [(x * u_size + m, y * u_size + m),
                                 ((x + 1) * u_size - m, y * u_size + m),
                                 ((x + 1) * u_size - m, (y + 1) * u_size - m),
                                 (x * u_size + m, (y + 1) * u_size - m)]
                    outline = rendering.make_polygon(v_outline, False)
                    outline.set_linewidth(3)
                    # the goal
                    if self._is_end_state(x, y):
                        # give end state cell a red .
                        g = [(x * 40 + 12, y * 40 + 12),
                             ((x + 1) * 40 - 12, y * 40 + 12),
                             ((x + 1) * 40 - 12, (y + 1) * 40 - 12),
                             (x * 40 + 12, (y + 1) * 40 - 12)]
                        goal_rect = rendering.FilledPolygon(g)
                        goal_rect.set_color(1, 0, 0)
                        outline.set_color(1, 0, 0)
                        self.viewer.add_geom(goal_rect)
                        self.viewer.add_geom(outline)
                    # agent start green colour
                    for start in self.starts:
                        if start[0] == x and start[1] == y:
                            outline.set_color(0.0, 1.0, 0.0)
                            self.viewer.add_geom(outline)
                    # obstacle cells are with gray color
                    if self.grids.get_type(x, y) == 0:
                        rect.set_color(0.3, 0.3, 0.3)
                    else:
                        pass
            # draw agent
            self.agent = rendering.make_circle(u_size / 4, 30, True)
            self.agent.set_color(0.0, 1.0, 0.0)
            self.viewer.add_geom(self.agent)
            self.agent_trans = rendering.Transform()
            self.agent.add_attr(self.agent_trans)
        # update position of an agent
        x, y = self.agent_states
        self.agent_trans.set_translation(
            (x + 0.5) * u_size, (y + 0.5) * u_size)
        return self.viewer.render(return_rgb_array=mode == 'rgb_array')


def maze_grid(starts, end, n_height, n_width, obstacles):
    """
    n_height*n_width maze environment
    @param starts:
    @param end:
    @param n_height:
    @param n_width:
    @param obstacles:
    @return: the RL openAI environment
    """
    env = GridWorldEnv(n_width,
                       n_height,
                       u_size=40,
                       default_reward=0,
                       default_type=1)
    env.starts = starts
    env.end = end
    env.obstacles = obstacles
    # use env.types describe the obstacle (x, y, 0)
    for obs in obstacles:
        obs_list = list(obs)
        obs_list.append(0)
        env.types.append(tuple(obs_list))   # 0 denotes the obstacle
    env.refresh_setting()
    return env


if __name__ == "__main__":
    starts = [(7, 9), (8, 2), (2, 8), (2, 0)]
    end = (4, 4)
    maze_length, maze_width = 10, 10
    obstacles = [(3, 7), (4, 7), (6, 7), (7, 7), (8, 7)]
    env = maze_grid(starts, end, maze_length, maze_width, obstacles)  # maze
    env.render()
    input('123')
