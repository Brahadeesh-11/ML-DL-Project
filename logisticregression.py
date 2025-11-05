from sklearn.linear_model import LogisticRegression
import numpy as np
clf = LogisticRegression()
X_bin = np.array([[45], [50], [55], [60], [65], [70], [90]])
y_bin = np.array([0, 0, 0, 1, 1, 1, 1]) 
clf.fit(X_bin, y_bin)
print("Model Score:", clf.score(X_bin, y_bin))