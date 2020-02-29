import numpy as np
from collections import deque
d = deque(maxlen=3)

a = np.array([[[1], [1]], [[2], [2]]]) #wo
b = np.array([[[3], [3]], [[4], [4]]]) # shi
c = np.array([[[5], [5]], [[6], [6]]]) # haoren

if (a == b).all():
    print('123')
else:
    print('1')

if a == 0:
    print('123')

