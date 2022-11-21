import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import random

# Data generation
size = 30
data = []
columns = ['Study Time', 'Scores']

# def dataGeneration():
    
#     studytime = np.random.randint(low=20, high=80, size=size)
#     print(f'Studytime => {studytime}')

#     value = np.random.random(size=size)
#     print(f'Values => {value}')

#     scores = np.array([])
#     for i in range(size):
#         x = studytime[i]*(1+value[i])
#         if x > 100:
#             scores = np.append(scores, [random.uniform(90, 100)])
#         else:
#             scores = np.append(scores, [x])
#     print(f'Scores => {scores}')

#     data = [[studytime[x], scores[x]] for x in range(size)]
#     return data

def gradient_descent(m_now, b_now, points, L):
    m_gradient = 0
    b_gradient = 0

    n = len(points)

    for i in range(n):
        x = points.iloc[i]['Study Time']
        y = points.iloc[i]['Scores']

        m_gradient += -(2/n)*x*(y-(m_now*x+b_now))
        b_gradient += -(2/n)*(y-(m_now*x+b_now))

        # time.sleep(0.1)
    
    m = m_now - m_gradient*L
    b = b_now - b_gradient*L
    return m,b

# Main method
df = pd.read_csv('./studyscore.csv')
print(df)

# weight of feature
m = 0
# bias
b = 0 
# Learning rate
L = 0.001
epoch = 100

for i in range(epoch):
    if i%20==0:
        print(f'Epoch: {i}')
    m,b = gradient_descent(m, b, df, L)

print(m,b)

plt.scatter(df['Study Time'], df['Scores'], color="black")
# plt.plot(list(range(20, 80)), [m*x-b for x in range(20, 80)])
plt.plot(df['Study Time'], [m*x+b for x in df['Study Time']])
plt.xlabel('Study Time')
plt.ylabel('Scores')
plt.title('Study time vs Score')
plt.show()