# %%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn import metrics
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import Lasso

# %%
df = pd.read_csv("./ENB2012_data.csv")
df.head()

# %%
df.info()

# %%
df.describe().T

# %%
corr = df.corr()
plt.figure(figsize=(16, 9))
sns.heatmap(corr, annot=True, cmap="jet")

# %%
X = df.drop(["Y1", "Y2"], axis=1)
y = df[["Y1", "Y2"]]

# %%
# Train Val Split
x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# %%
x_train


# %%
def evaluate(model):
    preds = model.predict(x_test)
    print("Testing error:")
    print(metrics.mean_squared_error(y_test, preds))
    print("Training error:")
    tpreds = model.predict(x_train)
    print(metrics.mean_squared_error(y_train, tpreds))
    plt.scatter(y_train.iloc[:, 0], tpreds[:, 0])
    plt.xlabel("True Y1")
    plt.ylabel("Predicted Y1")
    plt.title("Scatter plot of True Y1 vs Predicted Y1 (Training)")
    z1 = np.polyfit(y_train.iloc[:, 0], tpreds[:, 0], 1)
    p1 = np.poly1d(z1)
    plt.plot(y_train.iloc[:, 0], p1(y_train.iloc[:, 0]), color="red")
    plt.show()

    plt.scatter(y_train.iloc[:, 1], tpreds[:, 1])
    plt.xlabel("True Y2")
    plt.ylabel("Predicted Y2")
    plt.title("Scatter plot of True Y2 vs Predicted Y2 (Training)")
    z2 = np.polyfit(y_train.iloc[:, 1], tpreds[:, 1], 1)
    p2 = np.poly1d(z2)
    plt.plot(y_train.iloc[:, 1], p2(y_train.iloc[:, 1]), color="red")
    plt.show()


# %%
ridge = MultiOutputRegressor(Ridge(random_state=42))
ridge.fit(x_train, y_train)
evaluate(ridge)

# %%
lasso = MultiOutputRegressor(Lasso(random_state=42))
lasso.fit(x_train, y_train)
evaluate(lasso)

# %%
lr = MultiOutputRegressor(LinearRegression())
lr.fit(x_train, y_train)
evaluate(lr)
