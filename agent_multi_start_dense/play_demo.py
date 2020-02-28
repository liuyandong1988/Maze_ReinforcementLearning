import gym
import DQN_agent as dqn
import numpy as np
import time
from maze_env import maze_grid
from main_maze import generate_obstacles
from collections import deque

def main(starts, end, n_height, n_width, obstacles):
    # --- env
    start = starts[0]  # choose the start
    env = maze_grid(starts, end, n_height, n_width, obstacles)  # maze
    # env.render()
    # input('123')
    # --- model
    model = dqn.dqn_model((9, ), (25, ))
    # load the weight
    weight_file = 'dense_model.h5'
    model.load_weights(weight_file)
    # --- observation the new states
    goal_states, agent_states = env.reset(start, test=True)  # get the initial states
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
        env.render()
        steps += 1
        total_reward += r1
        if is_done:
            print('Arrive at the goal ...')
            break
        if steps > 20:
            print("Cannot arrive the goal ...")
            break
        env.render()
        time.sleep(0.3)
        # update states
        s1 = np.reshape(s1, [25, ])
        s1 = s1[np.newaxis, :]
        s0 = [goal_states, s1]
    print('The reward: %d, the agent path distance is %d step.' % (total_reward, steps))
    input('Stop')


if __name__ == "__main__":
    # choose the different instance

    # experiment 2
    # starts = [(0, 0), (8, 1), (0, 8), (8, 9)]
    starts = [(0, 8)]
    end = (4, 4)
    seed_num = 2
    maze_width = 10
    maze_length = 10
    obstacles = generate_obstacles(starts, end, seed_num, grid_num=maze_length * maze_width, percent=0.2)
    main(starts, end, maze_length, maze_width, obstacles)
