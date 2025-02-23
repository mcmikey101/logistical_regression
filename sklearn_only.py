from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer

X, y = load_breast_cancer(return_X_y=True)
log_r = LogisticRegression().fit(X, y)
print(log_r.score(X, y) * 100)