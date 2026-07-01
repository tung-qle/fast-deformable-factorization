import numpy as np

def ALS_rankone(A, x0, y0, max_iter = 100):
    x = x0
    y = y0
    for i in range(max_iter):
        y = np.dot(A.T, x) / (np.linalg.norm(x) ** 2)
        x = np.dot(A, y) / (np.linalg.norm(y) ** 2)
        print(i, np.linalg.norm(A - np.tensordot(x, y, axes = 0)))
    return x, y

if __name__ == "__main__":
    size1 = 4
    size2 = 3
    x = np.random.randn(size1)
    y = np.random.randn(size2)
    x0 = np.random.randn(size1)
    y0 = np.random.randn(size2)

    A = np.tensordot(x, y, axes = 0)
    x, y = ALS_rankone(A, x0, y0, max_iter = 100)
    print(np.linalg.norm(A - np.tensordot(x, y, axes = 0)))