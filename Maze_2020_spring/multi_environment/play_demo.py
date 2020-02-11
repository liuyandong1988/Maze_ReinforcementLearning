import DQN_agent as dqn
import numpy as np
import time
from maze_env import maze_grid
from main_maze import generate_obstacles

def main(start, ends, goal, n_height, n_width, obstacles, weight_file):
    # env
    env = maze_grid(start, ends, n_height, n_width, obstacles)  # maze
    env.render()
    # input('123')
    # model
    input_dim = env.observation_space.n
    hidden_dim = 256
    output_dim = env.action_space.n
    model = dqn.dqn_model(input_dim, hidden_dim, output_dim)
    model.load_weights(weight_file)

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
    """
    1. generate the environment, start/end
    2. randomly generate the obstacles, seed_num=3
    3. load model weight file
    """
    # choose the different instance
    start = (0, 1)
    ends = [(9, 8)]  # many ends
    target_goal = ends[0]
    maze_length = 10
    maze_width = 10
    seed_num = 3
    obstacles = generate_obstacles(start, ends, seed_num, grid_num=maze_length * maze_width, percent=0.2)
    # load the weight
    weight_file = 'multi_envs.h5'
    main(start, ends, target_goal, maze_length, maze_width, obstacles, weight_file)