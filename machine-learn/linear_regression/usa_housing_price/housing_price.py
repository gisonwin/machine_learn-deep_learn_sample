import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('../USA_Housing.csv')
print(data.head())

figure = plt.figure(figsize=(10, 10))
fig1 = plt.subplot(2, 3, 1)
# plt.scatter(data['SalePrice'],data['SalePrice'])
