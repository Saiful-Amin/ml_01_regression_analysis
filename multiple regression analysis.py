import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model,preprocessing

data = pd.read_csv("Metro_Interstate_Traffic_Volume.csv")

data = data[["temp", "rain_1h", "snow_1h", "clouds_all", "Hour", "traffic_volume"]]

estimate = "traffic_volume"

x = np.array(data.drop([estimate], 1))
y = np.array(data[estimate])

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size = 0.2)

linear = linear_model.LinearRegression()

linear.fit(x_train, y_train)
acc = linear.score(x_test, y_test)
print(acc)

print('Coefficients: \n', linear.coef_)
print('Intercept: \n', linear.intercept_)


