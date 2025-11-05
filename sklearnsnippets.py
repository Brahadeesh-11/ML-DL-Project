import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score

# 1️⃣ Create dataset for classification
X, y = make_classification(
    n_samples=200, n_features=5, n_informative=3,
    n_redundant=0, n_clusters_per_class=1, random_state=42
)

# 2️⃣ Split dataset for KNN training/testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# 3️⃣ Apply PCA (reduce to 2 features for plotting)
pca = PCA(n_components=2)
X_2d = pca.fit_transform(X)

# 4️⃣ KMeans clustering
km = KMeans(n_clusters=3, random_state=0)
clusters = km.fit_predict(X_2d)

# 5️⃣ Train KNN classifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# 6️⃣ Test KNN model
y_pred = knn.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"KNN Accuracy: {acc:.2f}")

# 7️⃣ Plot PCA + Clusters
plt.figure(figsize=(7,5))
plt.scatter(X_2d[:,0], X_2d[:,1], c=clusters, cmap='viridis')
plt.title("KMeans Clustering after PCA")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()

# Test a new sample (random example)
test_sample = np.array([[0.4, -1.2, 0.5, 1.0, -0.6]])
prediction = knn.predict(test_sample)
print("Prediction for new sample:", prediction)
