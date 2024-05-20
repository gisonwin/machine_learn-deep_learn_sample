# 画图对比y,y',可视化模型
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np

# 散点图

data = pd.read_csv('train.csv')
print(data.head())
print(type(data), data.shape)
# print(data.columns.tolist())
# # assume data
x1 = data.loc[:, 'x']
y1 = data.loc[:, 'y']
print(x1, y1)

# show data
plt.figure()
plt.scatter(x1, y1)
plt.show()

# setup linear regression model
lr_model = LinearRegression()
x = np.array(x1).reshape(-1, 1)
y = np.array(y1).reshape(-1, 1)
lr_model.fit(x, y)

y_pred = lr_model.predict(x)
print(y_pred, y)

y_3 = lr_model.predict([[3.5]])
print(y_3)

# 评估模型的表象
a = lr_model.coef_
b = lr_model.intercept_
print(a, b)
from sklearn.metrics import mean_squared_error, r2_score

mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)
print(mse, r2)

plt.figure()
plt.scatter(y, y_pred)
plt.show()
