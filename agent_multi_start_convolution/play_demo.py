import gym
import DQN_agent as dqn
import numpy as np
import time
from maze_env import maze_grid
from main_maze import generate_obstacles
from collections import deque

def main(starts, end, n_height, n_width, obstacles, weight_file):
    # --- env
    start = starts[0]  # choose the start
    env = maze_grid(starts, end, n_height, n_width, obstacles)  # maze
    env.render()
    # input('123')
    # --- model
    model = dqn.dqn_model((3, 3, 1), (5, 5, 3))
    # load the weight
    model.load_weights(weight_file)
    # --- observation the new states
    # --- initial agent states
    agent_init_states = np.ones((5, 5, 1))
    agent_3_states = deque(maxlen=3)
    # 3 continue state
    for _ in range(3):
        agent_3_states.append(agent_init_states)
    goal_states, agent_states = env.reset(
        start, test=True)  # get the initial states
    # --- observation the new states
    goal_states = np.reshape(goal_states, [3, 3, 1])
    goal_states = goal_states[np.newaxis, :]
    agent_states = np.reshape(agent_states, [5, 5, 1])
    agent_3_states.append(agent_states)
    # reshape the dim [3, 5, 5, 1] --> [1, 5, 5, 3]
    agent_s0 = np.concatenate((agent_3_states[0], agent_3_states[1], agent_3_states[2]), axis=2)
    agent_s0 = agent_s0[np.newaxis, :]

    # --- show the demo
    total_reward = 0
    is_done = False
    steps = 0
    while True:
        s0 = [goal_states, agent_s0]
        a0 = int(np.argmax(model.predict(s0)[0]))
        s1, r1, is_done, info = env.step(a0)
        env.render()
        steps += 1
        total_reward += r1
        if is_done:
            print('Arrive at the goal ...')
            break
        env.render()
        time.sleep(0.3)
        # update states
        s1 = np.reshape(s1, [5, 5, 1])
        agent_3_states.append(s1)
        # reshape the dim [3, 5, 5, 1] --> [1, 5, 5, 3]
        agent_s1 = np.concatenate((agent_3_states[0], agent_3_states[1], agent_3_states[2]), axis=2)
        agent_s1 = agent_s1[np.newaxis, :]
        agent_s0 = agent_s1

    print(
        'The reward: %d, the agent path distance is %d step.' %
        (total_reward, steps))
    time.sleep(1)


if __name__ == "__main__":
    # choose the different instance
    # experiment 1
    # starts = [(0, 2), (2, 8), (8, 1), (9, 9)]
    starts = [(2, 8)]
    end = (4, 4)
    maze_length = 10
    maze_width = 10
    seed_num = 1
    weight_file = "convolution_model.h5"
    obstacles = generate_obstacles(
        starts,
        end,
        seed_num,
        grid_num=maze_length *
        maze_width,
        percent=0.2)
    main(starts, end, maze_length, maze_width, obstacles, weight_file)


