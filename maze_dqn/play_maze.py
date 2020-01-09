import time, gym
import numpy as np
from keras.models import load_model
from maze_env import maze_grid


# choose the different instance
# start = (2, 9)
# ends = [(5, 4)]  # many ends
# instance 1
# obstacles = [(4, 2), (4, 3), (4, 4), (4, 5), (4, 6), (1, 7), (2, 7), (3, 7), (4, 7), (6, 7), (7, 7), (8, 7)]  # 1

# instance 2
# obstacles = [(4, 2), (4, 3), (4, 4), (4, 5), (4, 6), (1, 7), (2, 7), (3, 7), (4, 7), (5, 7), (6, 7), (7, 7), (8, 7)] #2

# instance difficult
start = (5, 9)
ends = [(4, 5)]
obstacles = [(2, 7), (3, 7), (4, 7), (5, 7), (6, 7), (7, 7), (3, 5), (6, 5), (2, 3), (3, 3), (6, 3), (7, 3)]

env = maze_grid(start, ends, obstacles)
model = load_model('./model3/dqn_maze_300.h5')

score_list = list()
episode = 10

# run 10 times
for i in range(episode):
    state = env.reset()
    score = 0
    while True:
        env.render()
        time.sleep(0.05)
        action = np.argmax(model.predict(np.array([state]))[0])
        state, reward, done, _ = env.step(action)
        score += reward
        if done:
            print('The {} time, score:{}'.format(i+1, score))
            score_list.append(score)
            time.sleep(0.5)
            break
print("The {} episode, average score is {:.2f}. ".format(episode, np.mean(score_list)))
env.close()