import keras
from optree import layers
import numpy as np
import tensorflow

# Create sample regression dataset
X = np.array([1,2,3,4,5], dtype=float)
y = np.array([2,4,6,8,10], dtype=float)

# Build model
model = keras.Sequential([
    layers.Dense(1, input_shape=(1,))  # type: ignore
])

model.compile(optimizer='sgd', loss='mse')

# Train
model.fit(X.reshape(-1,1), y, epochs=100, verbose=1) # type: ignore

# Test prediction
print(model.predict([[6]]))
