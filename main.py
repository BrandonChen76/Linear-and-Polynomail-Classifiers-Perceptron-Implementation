import numpy as np
import pandas as pd

#data frame of the example set
data = {
    'name' : ["e1", "e2", "e3"],
    'x1' : [1, 1, 0],
    'x2' : [0, 1, 0],
    'c' : [0, 1, 0],
}
df = pd.DataFrame(data)

#initializing weights including w0 -> 3
w = np.random.rand(3)

print("Initial Weights: ", w)

#learning rate
l = 0.4

#algorithm needs to continue to check all until all correct
correctCount = 0

#record index
i = 0

#loop through the dataframe until classifier is correct for all rows
while correctCount < df.shape[0]:
    #check row
    curr = df.iloc[i]

    #get the prediction based on weight
    prediction = -1
    if (w[0] + w[1] * curr['x1'] + w[2] * curr['x2']) > 0:
        prediction = 1
    else:
        prediction = 0

    #compare to actual and if it is wrong, update weights
    if(prediction != curr['c']):
        w[0] = w[0] + (l * (curr['c'] - prediction))
        w[1] = w[1] + (l * (curr['c'] - prediction) * curr['x1'])
        w[2] = w[2] + (l * (curr['c'] - prediction) * curr['x2'])
        correctCount = 0
    else:
        correctCount += 1
    
    if i == df.shape[0] - 1:
        i = -1
    i += 1

#outputs
print("Updated Weights: ", w)

#print the predictions from the correct weights
for i, row in df.iterrows():
    print("")
    print(row['name'], "\nActual:", row['c'])
    print("Prediction:", w[0] + w[1] * row['x1'] + w[2] * row['x2'])
    
