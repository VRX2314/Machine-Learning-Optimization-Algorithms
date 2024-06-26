# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm

# %%
# Loading the data
df = pd.read_csv("./data/data.csv")

# Dropping unnamed column
df = df.drop(df.columns[-1], axis=1)
df.head()

# %%
df.info()

# No NULL values present

# %%
df.describe().T

# No missing values (Usually stated as -1)

# %%
# Checking data distribution and balance

print(df["diagnosis"].value_counts())
_ = sns.countplot(df, x="diagnosis", hue="diagnosis")
# Data seems to be imbalanced

# %%
# Encoding target to 0 and 1
df["diagnosis"] = df["diagnosis"].replace({"M": 1, "B": 0})

# %%
# Understanding correleation
corr = df.corr()
corr

# %%
plt.figure(figsize=(20, 20))
sns.heatmap(corr, annot=True, cmap="jet")

# %%
# Using data with high correlation

threshold = 0.75
filtre = np.abs(corr["diagnosis"] > threshold)
corr_features = corr.columns[filtre].tolist()
plt.figure(figsize=(10, 8))
sns.clustermap(df[corr_features].corr(), annot=True, fmt=".2f")
plt.title(
    "\n                               Correlation Between Features with Cor Thresgold [0.75]\n",
    fontsize=20,
)
plt.show()

# %%
sns.pairplot(df[corr_features], diag_kind="kde", markers="*", hue="diagnosis")
plt.show()

# %%
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Scaling data using standard scaler
scaler = StandardScaler()

X = df.drop(["id", "diagnosis"], axis=1)
y = df["diagnosis"]

X_norm = scaler.fit_transform(X)

# Making a train test split
X_train, X_test, y_train, y_test = train_test_split(
    X_norm, y, test_size=0.2, random_state=10
)

# %%
import torch
import torch.nn as nn
from torch.optim import SGD

torch.manual_seed(0)
device = torch.device("mps")

# %%
# Implementing Logistic Regression from scratch


class LogisticRegression(nn.Module):
    def __init__(self, n_in, n_out):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(n_in, n_out)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.sigmoid(self.linear(x))
        return x


# %%
# Init model

in_features = X_train.shape[1]
out_features = 1

model = LogisticRegression(in_features, out_features)

# %%
# Making data tensors

X_train_tensor = torch.tensor(X_train).float()
X_test_tensor = torch.tensor(X_test).float()
y_train_tensor = torch.tensor(np.array(y_train)).float()
y_test_tensor = torch.tensor(np.array(y_test)).float()

# %%
# Initialize parameters

# Using a very heavy L2 penalty as model tends to overfit
optimizer = SGD(model.parameters(), 0.01, 0.9, weight_decay=0.1)
criterion = nn.BCELoss()
epochs = 1000
batch_size = 32

# %%
# Making a train loop

loss_log = []
acc_log = []
val_loss_log = []
val_acc_log = []

# Setting a big number
best_acc = -1

for epoch in tqdm(range(epochs)):
    running_loss = []
    running_acc = []
    running_val_loss = []
    running_val_acc = []

    for batch in range(0, len(X_train), batch_size):
        optimizer.zero_grad()
        data = X_train_tensor[batch : batch + batch_size]
        target = y_train_tensor[batch : batch + batch_size]

        data.to(device)
        target.to(device)

        pred = model(data)
        abs_pred = torch.where(pred > 0.5, 1, 0).detach().cpu().numpy()
        abs_target = target.unsqueeze(1).detach().cpu().numpy()

        loss = criterion(pred, target.unsqueeze(1))
        acc = accuracy_score(abs_target, abs_pred)
        loss.backward()
        optimizer.step()

        running_loss.append(loss.item())
        running_acc.append(acc)

    with torch.no_grad():
        test_data = X_test_tensor
        test_target = y_test_tensor

        test_data.to(device)
        test_target.to(device)

        pred = model(test_data)
        abs_pred = torch.where(pred > 0.5, 1, 0).detach().cpu().numpy()
        abs_target = test_target.unsqueeze(1).detach().cpu().numpy()

        val_loss = criterion(pred, test_target.unsqueeze(1))
        val_acc = accuracy_score(abs_target, abs_pred)

        running_val_loss.append(loss.item())
        running_val_acc.append(val_acc)

    # Saving models with best accuracy
    if np.array(running_val_acc).mean() > best_acc:
        torch.save(model, "./chkpt_1.pth")

    loss_log.append(np.array(running_loss).mean())
    acc_log.append(np.array(acc).mean())
    val_loss_log.append(np.array(running_val_loss).mean())
    val_acc_log.append(np.array(running_val_acc).mean())

# %%
best_model = torch.load("./chkpt_1.pth")

# %%
val_pred = best_model(X_test_tensor)
print(classification_report(y_test_tensor, torch.where(val_pred > 0.5, 1, 0).numpy()))

# %%
plt.plot(acc_log)
plt.plot(val_acc_log)
plt.legend(["Train Accuracy", "Val Accuracy"])
plt.show()

plt.plot(loss_log)
plt.plot(val_loss_log)
plt.legend(["Train Loss", "Val Loss"])

# %%
sns.heatmap(
    confusion_matrix(y_test_tensor, torch.where(val_pred > 0.5, 1, 0).numpy()),
    annot=True,
)
