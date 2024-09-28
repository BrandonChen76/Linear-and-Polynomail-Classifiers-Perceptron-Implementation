import numpy as np
import pandas as pd

#data frame of the example set
data = {
    'name' : ["e1", "e2", "e3"],
    'x1' : [1, 1, 0],
    'x2' : [0, 1, 0],
    'c' : [0, 1, 0],
}

#initializing weights including w0
weights = np.random.rand(3)

print(weights)