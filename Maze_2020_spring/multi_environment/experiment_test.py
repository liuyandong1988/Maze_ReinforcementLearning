import gym
import DQN_agent as dqn
import numpy as np
import time
from maze_env import maze_grid
from main_maze import generate_obstacles

def main(start, ends, n_height, n_width, obstacles):
    # env
    env = maze_grid(start, ends, n_height, n_width, obstacles)  # maze
    env.render()
    # model
    input_dim = env.observation_space.n
    hidden_dim = 256
    output_dim = env.action_space.n
    model = dqn.dqn_model(input_dim, hidden_dim, output_dim)
    # load the weight
    weight_file = 'single_agent_2_goals.h5'
    model.load_weights(weight_file)
    reward_record = dict()
    for i, goal in enumerate(ends):
        each_goal_record = list()
        print('Goal', goal)
        # show the demo
        for _ in range(10):
            states = list()
            env.reset(states, show=goal)
            env.render()
            s0 = np.reshape(states, [1, input_dim])
            total_reward = 0
            is_done = False
            steps = 0
            while not is_done:
                a0 = int(np.argmax(model.predict(s0)[0]))
                s1, r1, is_done, info = env.step(a0)
                env.render()
                time.sleep(0.1)
                s1 = np.reshape(s1, [1, input_dim])
                s0 = s1
                total_reward += r1
                steps += 1
            print('The reward: %d, the agent path distance is %d step.' % (total_reward, steps))
            each_goal_record.append(total_reward)
            time.sleep(0.5)
        print('10 times mean reward:%f reward: %s' % (np.mean(each_goal_record), each_goal_record))
        reward_record[i + 1] = each_goal_record

    print(reward_record)


if __name__ == "__main__":
    # choose the different instance
    start = (2, 9)
    ends = [(0, 9), (9, 2)] # many ends
    maze_length = 10
    maze_width = 10
    seed_num = 2
    obstacles = generate_obstacles(start, ends, seed_num, grid_num=maze_length * maze_width, percent=0.2)
    main(start, ends, maze_length, maze_width, obstacles)