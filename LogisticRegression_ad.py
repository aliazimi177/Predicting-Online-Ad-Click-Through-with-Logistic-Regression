from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder
import timeit 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier

def sigmoid(input):
    return 1.0 / (1 + np.exp(-input))

def compute_prediction(X, weights):
    z = np.dot(X, weights)
    predictions = sigmoid(z)
    return predictions
def update_weights_gd(X_train, y_train, weights,learning_rate):
    predictions = compute_prediction(X_train, weights)
    weights_delta = np.dot(X_train.T, y_train - predictions)
    m = y_train.shape[0]
    weights += learning_rate / float(m) * weights_delta
    return weights
def compute_cost(X, y, weights):
    predictions = compute_prediction(X, weights)
    cost = np.mean(-y * np.log(predictions)- (1 - y) * np.log(1 - predictions))
    return cost
def train_logistic_regression(X_train, y_train, max_iter,
learning_rate, fit_intercept=False):
    if fit_intercept:
        intercept = np.ones((X_train.shape[0], 1))
        X_train = np.hstack((intercept, X_train))
    weights = np.zeros(X_train.shape[1])
    for iteration in range(max_iter):
        weights = update_weights_gd(X_train, y_train,weights, learning_rate)
        if iteration % 100 == 0:
            print(compute_cost(X_train, y_train, weights))
    return weights
def predict(X, weights):
    if X.shape[1] == weights.shape[0] - 1:
        intercept = np.ones((X.shape[0], 1))
        X = np.hstack((intercept, X))
    return compute_prediction(X, weights)
n_rows = 300000
df = pd.read_csv("train", nrows=n_rows)
X = df.drop(['click', 'id', 'hour', 'device_id', 'device_ip'],
axis=1).values
Y = df['click'].values
n_train = 10000
X_train = X[:n_train]
Y_train = Y[:n_train]
X_test = X[n_train:]
Y_test = Y[n_train:]
enc = OneHotEncoder(handle_unknown='ignore')
X_train_enc = enc.fit_transform(X_train)
X_test_enc = enc.transform(X_test)
start_time = timeit.default_timer()
weights = train_logistic_regression(X_train_enc.toarray(),Y_train, max_iter=10000, learning_rate=0.01,fit_intercept=True)
print(f"--- {(timeit.default_timer() - start_time)}.3fs seconds ---")
def update_weights_sgd(X_train, y_train, weights,learning_rate):
    for X_each, y_each in zip(X_train, y_train):
        prediction = compute_prediction(X_each, weights)
        weights_delta = X_each.T * (y_each - prediction)
        weights += learning_rate * weights_delta
    return weights
def train_logistic_regression_sgd(X_train, y_train, max_iter,learning_rate, fit_intercept=False):
    if fit_intercept:
        intercept = np.ones((X_train.shape[0], 1))
        X_train = np.hstack((intercept, X_train))
    weights = np.zeros(X_train.shape[1])
    for iteration in range(max_iter):
        weights = update_weights_sgd(X_train, y_train, weights,learning_rate)
        if iteration % 2 == 0:
            print(compute_cost(X_train, y_train, weights))
    return weights
sgd_lr = SGDClassifier(loss='log', penalty=None,
fit_intercept=True, max_iter=10,
learning_rate='constant', eta0=0.01)
sgd_lr.fit(X_train_enc.toarray(), Y_train)
pred = sgd_lr.predict_proba(X_test_enc.toarray())[:, 1]
print(f'Training samples: {n_train}, AUC on testing set: {roc_auc_score(Y_test, pred):.3f}')
sgd_lr_l1 = SGDClassifier(loss='log', penalty='l1', alpha=0.0001,
fit_intercept=True, max_iter=10,
learning_rate='constant', eta0=0.01)
sgd_lr_l1.fit(X_train_enc.toarray(), Y_train)
