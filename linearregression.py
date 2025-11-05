from sklearn.linear_model import LinearRegression
import numpy as np
model = LinearRegression()
X = np.array([1.0,2.0,3.0])
y = np.array([2.0,3.0,5.0])
model.fit(X.reshape(-1,1), y)
print(model.coef_, model.intercept_)