import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

torch.manual_seed(42)
np.random.seed(42)

def load_data(file_path):
    try:
        data = pd.read_csv(file_path)
        x = data.iloc[:, 0].values.astype(np.float32)
        y = data.iloc[:, 1].values.astype(np.float32)
        x_tensor = torch.FloatTensor(x).view(-1, 1)
        y_tensor = torch.FloatTensor(y).view(-1, 1)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
        x_train = torch.FloatTensor(x_train).view(-1, 1)
        x_test = torch.FloatTensor(x_test).view(-1, 1)
        y_train = torch.FloatTensor(y_train).view(-1, 1)
        y_test = torch.FloatTensor(y_test).view(-1, 1)
        return x_train, x_test, y_train, y_test, x_tensor, y_tensor
    except:
        return None

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        return self.net(x)

def train():
    result = load_data("data.csv")
    if result is None:
        print("使用示例数据")
        x = np.linspace(-3, 3, 300)
        y = np.sin(x) + 0.1 * np.random.randn(300)
        x_tensor = torch.FloatTensor(x).view(-1, 1)
        y_tensor = torch.FloatTensor(y).view(-1, 1)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
        x_train = torch.FloatTensor(x_train).view(-1, 1)
        x_test = torch.FloatTensor(x_test).view(-1, 1)
        y_train = torch.FloatTensor(y_train).view(-1, 1)
        y_test = torch.FloatTensor(y_test).view(-1, 1)
    else:
        x_train, x_test, y_train, y_test, x_tensor, y_tensor = result

    model = MLP()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1)

    pred_dict = {10: None, 100: None, 1000: None}

    for epoch in range(1000):
        pred = model(x_train)
        loss = criterion(pred, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch + 1 in [10, 100, 1000]:
            with torch.no_grad():
                pred_dict[epoch + 1] = model(x_tensor).numpy().copy()

        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch + 1}, Loss: {loss.item():.6f}")

    # 绘图
    plt.figure(figsize=(15, 4))
    x_np = x_tensor.numpy().flatten()
    y_np = y_tensor.numpy().flatten()
    sort_idx = np.argsort(x_np)
    x_sorted = x_np[sort_idx]

    plt.subplot(1, 3, 1)
    plt.scatter(x_np, y_np, alpha=0.5, s=5, c='blue', label='真实数据')
    if pred_dict[10] is not None:
        plt.plot(x_sorted, pred_dict[10].flatten()[sort_idx], 'r-', linewidth=2, label='Epoch=10')
    plt.title('Epoch = 10')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 3, 2)
    plt.scatter(x_np, y_np, alpha=0.5, s=5, c='blue', label='真实数据')
    if pred_dict[100] is not None:
        plt.plot(x_sorted, pred_dict[100].flatten()[sort_idx], 'g-', linewidth=2, label='Epoch=100')
    plt.title('Epoch = 100')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 3, 3)
    plt.scatter(x_np, y_np, alpha=0.5, s=5, c='blue', label='真实数据')
    if pred_dict[1000] is not None:
        plt.plot(x_sorted, pred_dict[1000].flatten()[sort_idx], 'b-', linewidth=2, label='Epoch=1000')
    plt.title('Epoch = 1000')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('epoch_comparison.png', dpi=300)
    plt.show()
    print("图片已保存！")

if __name__ == "__main__":
    train()
