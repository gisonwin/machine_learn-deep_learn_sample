import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

data = pd.read_csv('../USA_Housing.csv')


# print(data.head(10))


def fig_show():
    figure = plt.figure(figsize=(10, 10))
    fig1 = plt.subplot(2, 3, 1)
    plt.scatter(data.loc[:, 'Avg. Area Income'], data.loc[:, 'Price'])
    plt.title("Price vs Avg. Area Income")

    fig2 = plt.subplot(2, 3, 2)
    plt.scatter(data.loc[:, 'Avg. Area House Age'], data.loc[:, 'Price'])
    plt.title("Price vs Avg. Area House Age")

    fig3 = plt.subplot(2, 3, 3)
    plt.scatter(data.loc[:, 'Avg. Area Number of Rooms'], data.loc[:, 'Price'])
    plt.title("Price vs Avg. Area Number of Rooms")

    fig4 = plt.subplot(2, 3, 4)
    plt.scatter(data.loc[:, 'Avg. Area Number of Bedrooms'], data.loc[:, 'Price'])
    plt.title("Price vs Avg. Area Number of Bedrooms")

    fig5 = plt.subplot(2, 3, 5)
    plt.scatter(data.loc[:, 'Area Population'], data.loc[:, 'Price'])
    plt.title("Price vs Population")

    plt.show()


# fig_show()
x = data.loc[:, 'Area Population']
y = data.loc[:, 'Price']
print(x, y)
x = np.array(x).reshape(-1, 1)
y = np.array(y).reshape(-1, 1)
# print(x.shape, y.shape)
LR1 = LinearRegression()
# train the model
LR1.fit(x, y)

y_predict_1 = LR1.predict(x)
print(y_predict_1)
