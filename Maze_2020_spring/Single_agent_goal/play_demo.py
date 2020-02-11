import gym
import DQN_agent as dqn
import numpy as np
import time
from maze_env import maze_grid
from main_maze import generate_obstacles

def main(start, ends, goal, n_height, n_width, obstacles):
    # env
    env = maze_grid(start, ends, n_height, n_width, obstacles)  # maze
    env.render()
    # input('123')
    # model
    input_dim = env.observation_space.n
    hidden_dim = 256
    output_dim = env.action_space.n
    model = dqn.dqn_model(input_dim, hidden_dim, output_dim)
    # load the weight
    weight_file = 'single_agent_5_goals.h5'
    model.load_weights(weight_file)
    reward_record = dict()

    print('Goal', goal)
    # show the demo
    states = list()
    env.reset(states, show=goal)
    s0 = np.reshape(states, [1, input_dim])
    total_reward = 0
    is_done = False
    steps = 0
    while not is_done:
        a0 = int(np.argmax(model.predict(s0)[0]))
        s1, r1, is_done, info = env.step(a0)
        env.render()
        time.sleep(0.3)
        s1 = np.reshape(s1, [1, input_dim])
        s0 = s1
        total_reward += r1
        steps += 1
        print('The reward: %d, the agent path distance is %d step.' % (total_reward, steps))
    time.sleep(1)


if __name__ == "__main__":
    # choose the different instance
    start = (4, 4)
    ends = [(7, 2), (4, 0), (0, 9), (9, 8), (5, 7)] # many ends
    target_goal = ends[4]
    maze_length = 10
    maze_width = 10
    seed_num = 1
    obstacles = generate_obstacles(start, ends, seed_num, grid_num=maze_length * maze_width, percent=0.2)
    main(start, ends, target_goal, maze_length, maze_width, obstacles)