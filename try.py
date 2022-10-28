import numpy as np

def haha(a):
    np.add.at(a, a[0], [1, 2])

a = np.array([
    [1, 2],
    [3, 4]
])

b = np.array([
    [2, 3],
    [4, 5]
])

print(a, b)
haha(a)
print(a, b)
haha(b)
print(a)