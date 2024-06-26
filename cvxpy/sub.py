# %%
import pandas as pd
import numpy as np

import warnings

import warnings

warnings.filterwarnings("ignore")

# %%
df=pd.read_csv("./salary_data.csv")
df.head()

# %%
df.info()

# %%
x = df['YearsExperience']
y = df['Salary']

# %%
def linear_regression(x, y):
    N = len(x)
    x_mean = x.mean()
    y_mean = y.mean()

    B1_num = ((x - x_mean) * (y - y_mean)).sum()
    B1_den = ((x - x_mean)**2).sum()
    B1 = B1_num / B1_den

    B0 = y_mean - (B1*x_mean)

    reg_line = 'y = {} + {}β'.format(B0, round(B1, 3))

    return (B0, B1, reg_line)

N = len(x)
x_mean = x.mean()
y_mean = y.mean()

B1_num = ((x - x_mean) * (y - y_mean)).sum()
B1_den = ((x - x_mean)**2).sum()
B1 = B1_num / B1_den

B0 = y_mean - (B1 * x_mean)

B0, B1, reg_line = linear_regression(x, y)
print('Regression Line: ', reg_line)

def predict(B0, B1, new_x):
    y = B0 + B1 * new_x
    return y

# %%
y = predict(B0,B1,x)
print(y)

df['predicted'] = y
n = len(y)
act=df["Salary"]
pred=df["predicted"]
mse = sum([(act[i] - pred[i])**2 for i in range(n)]) / n
print("MSE:", mse)

# %%
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

print(X_train.shape)
print(X_test.shape)

if X_train.ndim == 1:
    X_train = X_train.values.reshape(-1, 1)
if X_test.ndim == 1:
    X_test = X_test.values.reshape(-1, 1)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# LASSO

lasso_model = Lasso(alpha=1.0)
lasso_model.fit(X_train, y_train)
y_pred = lasso_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:",mse)
print("Coefficients:",lasso_model.coef_)
print("Intercept:",lasso_model.intercept_)

# RIDGE

ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train, y_train)
y_pred_ridge = ridge_model.predict(X_test)
mse_ridge = mean_squared_error(y_test, y_pred_ridge)
print("Mean Squared Error (Ridge):", mse_ridge)
print("Coefficients (Ridge):", ridge_model.coef_)
print("Intercept (Ridge):", ridge_model.intercept_)

# %%
import cvxpy as cp
import numpy as np

n = 10
np.random.seed(1)
A = np.random.randn(n, n)
x_star = np.random.randn(n)
b = A @ x_star
epsilon = 1e-2

x = cp.Variable(n)
mse = cp.sum_squares(A @ x - b)/n
problem = cp.Problem(cp.Minimize(cp.length(x)), [mse <= epsilon])
print("Is problem DQCP?: ", problem.is_dqcp())

problem.solve(qcp=True)
print("Found a solution, with length: ", problem.value)

print("MSE: ", mse.value)

print("x: ", x.value)

print("x_star: ", x_star)

# %%
import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt

def loss_fn(X, Y, beta):
    preds=X @ beta
    # Y=Y.reshape(-1,1)
    loss=cp.sum(cp.abs(preds-Y))/len(Y)
    return loss

def regularizer(beta, lambd):
    return (lambd / (2)) * cp.sum_squares(beta)

def objective_fn(X, Y, beta, lambd):
    return loss_fn(X, Y, beta) + regularizer(beta, lambd)

def mse(X, Y, beta):
    preds=X @ beta
    mse=cp.sum((preds-Y)**2)/len(Y)
    return mse

def generate_data(m=100, n=20, sigma=5):
    "Generates data matrix X and observations Y."
    np.random.seed(1)
    beta_star = np.random.randn(n)
    # Generate an ill-conditioned data matrix
    X = np.random.randn(m, n)
    # Corrupt the observations with additive Gaussian noise
    Y = X.dot(beta_star) + np.random.normal(0, sigma, size=m)
    return X, Y

m = 100
n = 20
sigma = 5

X, Y = generate_data(m, n, sigma)
X_train = X[:50, :]
Y_train = Y[:50]
X_test = X[50:, :]
Y_test = Y[50:]

beta = cp.Variable(n)
lambd = cp.Parameter(nonneg=True)
problem = cp.Problem(cp.Minimize(objective_fn(X_train, Y_train, beta, lambd)))

lambd_values = np.logspace(-2, 3, 50)
train_errors = []
test_errors = []
beta_values = []
for v in lambd_values:
    lambd.value = v
    problem.solve()
    train_errors.append(mse(X_train, Y_train, beta))
    test_errors.append(mse(X_test, Y_test, beta))
    beta_values.append(beta.value)

def plot_train_test_errors(train_errors, test_errors, lambd_values):
    plt.plot(lambd_values, train_errors, label="Train error")
    plt.plot(lambd_values, test_errors, label="Test error")
    plt.xscale("log")
    plt.legend(loc="upper left")
    plt.xlabel(r"$\lambda$", fontsize=16)
    plt.title("Mean Squared Error (MSE)")
    plt.show()

print(train_errors, test_errors, lambd_values)
train_errors_values = [train_error.value for train_error in train_errors]
test_errors_values = [test_error.value for test_error in test_errors]
plot_train_test_errors(train_errors_values, test_errors_values, lambd_values)

def plot_regularization_path(lambd_values, beta_values):
    num_coeffs = len(beta_values[0])
    for i in range(num_coeffs):
        plt.plot(lambd_values, [wi[i] for wi in beta_values])
    plt.xlabel(r"$\lambda$", fontsize=16)
    plt.xscale("log")
    plt.title("Regularization Path")
    plt.show()

plot_regularization_path(lambd_values, beta_values)


# %%
import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt

def loss_fn(X, Y, beta):
    preds=X @ beta
    # Y=Y.reshape(-1,1)
    loss=cp.sum(cp.abs(preds-Y))/len(Y)
    return loss

def regularizer(beta, lamd):
    return lambd*cp.norm1(beta)

def objective_fn(X, Y, beta, lambd):
    return loss_fn(X, Y, beta) + regularizer(beta, lambd)

def mse(X, Y, beta):
    preds=X @ beta
    mse=cp.sum((preds-Y)**2)/len(Y)
    return mse

def generate_data(m=100, n=20, sigma=5, density=0.2):
    "Generates data matrix X and observations Y."
    np.random.seed(1)
    beta_star = np.random.randn(n)
    idxs = np.random.choice(range(n), int((1-density)*n), replace=False)
    for idx in idxs:
        beta_star[idx] = 0
    X = np.random.randn(m,n)
    Y = X.dot(beta_star) + np.random.normal(0, sigma, size=m)
    return X, Y, beta_star

m = 100
n = 20
sigma = 5
density = 0.2

X, Y, _ = generate_data(m, n, sigma)
X_train = X[:50, :]
Y_train = Y[:50]
X_test = X[50:, :]
Y_test = Y[50:]

beta = cp.Variable(n)
lambd = cp.Parameter(nonneg=True)
problem = cp.Problem(cp.Minimize(objective_fn(X_train, Y_train, beta, lambd)))

lambd_values = np.logspace(-2, 3, 50)
train_errors = []
test_errors = []
beta_values = []
for v in lambd_values:
    lambd.value = v
    problem.solve()
    train_errors.append(mse(X_train, Y_train, beta))
    test_errors.append(mse(X_test, Y_test, beta))
    beta_values.append(beta.value)

def plot_train_test_errors(train_errors, test_errors, lambd_values):
    plt.plot(lambd_values, train_errors, label="Train error")
    plt.plot(lambd_values, test_errors, label="Test error")
    plt.xscale("log")
    plt.legend(loc="upper left")
    plt.xlabel(r"$\lambda$", fontsize=16)
    plt.title("Mean Squared Error (MSE)")
    plt.show()

print(train_errors, test_errors, lambd_values)
train_errors_values = [train_error.value for train_error in train_errors]
test_errors_values = [test_error.value for test_error in test_errors]
plot_train_test_errors(train_errors_values, test_errors_values, lambd_values)

def plot_regularization_path(lambd_values, beta_values):
    num_coeffs= len(beta_values[0])
    for i in range(num_coeffs):
        plt.plot(lambd_values, [wi[i] for wi in beta_values])
    plt.xlabel(r"$\lambda$", fontsize=16)
    plt.xscale("log")
    plt.title("Regularization Path")
    plt.show()

plot_regularization_path(lambd_values, beta_values)

print(beta_values)


