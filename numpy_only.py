import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)

n = X.shape[1]

X_train = np.c_[np.ones(X_train.shape[0]), X_train]
X_test = np.c_[np.ones(X_test.shape[0]), X_test]

theta = np.zeros(n + 1)

def sigmoid(n):
    return 1 / (1 + np.exp(-n))

def cost_function(X, theta, y):
    lil = 1e-7
    m = X.shape[0]
    h = sigmoid(np.matmul(X, theta))
    return (-1 / m) * np.sum(np.dot(y, np.log(h + lil)) + np.dot((1 - y), np.log(1 - h + lil)))

alpha = 0.000001
iterations = 5000
cost_history = []

for _ in range(iterations):
    m = X_train.shape[0]
    h = sigmoid(np.matmul(X_train, theta))
    grad = (alpha / m) * X_train.T.dot(h - y_train)
    theta -= grad
    cost_history.append(cost_function(X_train, theta, y_train))

def compute_confusion_matrix(true, pred):
  k = len(np.unique(true))
  result = np.zeros((k, k))

  for i in range(len(true)):
    result[true[i]][pred[i]] += 1

  return result

def predict(X, theta):
   return sigmoid(np.matmul(X, theta))

pred = np.round(predict(X_test, theta)).astype(np.int32)

con_matrix = compute_confusion_matrix(y_test, pred)

print('Accuracy: ', (con_matrix[0][0] + con_matrix[1][1]) / X_test.shape[0] * 100)

plt.plot([i for i in range(iterations)], cost_history)
plt.show()