import numpy as np


def train_linear(X, y, epochs=1000, lr=0.01):
    N = X.shape[0] # pyright: ignore[reportUndefinedVariable]
    w = 0.0
    b = 0.0
    for epoch in range(epochs):
        y_pred = w * X + b
        error = y_pred - y
        loss = np.mean(error**2)
        dw = (2.0/N) * np.sum(error * X)
        db = (2.0/N) * np.sum(error)
        w -= lr * dw
        b -= lr * db
        if epoch % (epochs//5) == 0:
            print(f"epoch {epoch}, loss={loss:.4f}, w={w:.4f}, b={b:.4f}")
    return w, b


if __name__ == '__main__':
    X = np.array([1.0,2.0,3.0])
    y = np.array([2.0,3.0,5.0])
    train_linear(X, y, epochs=200, lr=0.1)