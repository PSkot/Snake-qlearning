import numpy as np

# States:
    # Candy left/right
    # Candy above/below
    # Direction

# Rewards
    # Candy picked up: 50
    # Wall hit: -10
    # Tail hit: -10
    # Closer xaxis: 1
    # Further away xaxis: -1.5
    # Closer yaxis: 1
    # Further away yaxis: -1.5

states = np.zeros((4, 3, 3))
directions = [0, 0, 0, 1]
left_right = [0, 0, 1]
above_below = [0, 0, 1]
index_dir = np.argwhere(directions)[0][0]
index_lr = np.argwhere(left_right)[0][0]
index_ab = np.argwhere(above_below)[0][0]

states[index_dir, index_lr, index_ab] = 1
print(np.argwhere(states.ravel()))

print(index_dir*9+index_lr*3+index_ab)

print(np.zeros(range(10)))

x, y = 5, 4
print(x, y)


if -1:
    print("Hello")
