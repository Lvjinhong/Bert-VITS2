import numpy as np


t = np.arange(49)
t.resize(7, 7)
t
t[[1, 3], [2, 4]]

ax = np.array([1, 3])
ax.resize(2, 1)
t[ax,]
t[ax, :]
