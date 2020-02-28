from collections import deque
d=deque(maxlen=3)

import numpy as np
a = np.array([[[1], [1]], [[2], [2]]])
b = np.array([[[3], [3]], [[4], [4]]])
c = np.array([[[5], [5]], [[6], [6]]])
print(a.shape)
d.append(a)
d.append(b)
d.append(c)

print(d)
change_d = np.array(d)
print(change_d.shape)

# change_d = np.reshape(change_d, [2, 2, 3])
# print(change_d.shape)
# print(change_d)
#
#

r = np.concatenate((d[0], d[1], d[2]), axis=2)
print(r.shape)
print(r)
