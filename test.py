import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import time

data = pd.read_csv('./canada_per_capita_income.csv')
# print(data)
# plt.scatter(data.year, data.per_capita_income)
# plt.show()

# Mean Square error
def loss_function(m, b, points):
    total_loss = 0
    for i in range(len(points)):
        x = points.iloc[i].year
        y = points.iloc[i].per_capita_income
        total_loss += (y - (m*x+b))**2
    total_loss / float(len(points))

def gradient_descent(m_now, b_now, points, L):
    m_gradient = 0
    b_gradient = 0

    n = len(points)

    for i in range(n):
        x = points.iloc[i].year
        y = points.iloc[i].per_capita_income

        m_gradient += -(2/n)*x*(y-(m_now*x+b_now))
        b_gradient += -(2/n)*(y-(m_now*x+b_now))
        print(m_gradient, b_gradient)
        # time.sleep(0.1)
    
    m = m_now - m_gradient*L
    b = b_now - b_gradient*L
    return m,b

# weight of feature
m = 0
# bias
b = 0 
# Learning rate
L = 0.1
epoch = 1000

for i in range(epoch):
    if i%50==0:
        print(f'Epoch: {i}')
    m,b = gradient_descent(m, b, data, L)

print(m,b)
plt.scatter(data.year, data.per_capita_income, color="black")
plt.plot(list(range(1970, 2017)), [m*x+b for x in range(1970, 2017)], color="red")
plt.xlabel('Year')
plt.ylabel('Per Capita Income')
plt.title('Per capita income since 1970')
plt.show()