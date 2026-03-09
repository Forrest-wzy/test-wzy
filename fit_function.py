from operator import truediv, index

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torchvision
from fontTools.misc.iterTools import batched
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

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
    optimizer = optim.Adam(model.parameters(), lr=0.00001)

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
#任务三
#torchvision
from torchvision import datasets
from torchvision import transforms

transform = transforms.Compose([
    transforms.ToTensor(),  # 把图片从PIL格式转成PyTorch张量
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 归一化
])

# 2. 下载/加载训练集
train_set = datasets.CIFAR10(
    root='D:/pytorch/data',       # 存到当前目录的 data 文件夹
    train=True,          # 这是训练集
    download=False,
    transform=transform  # 应用预处理
)

# 3. 下载/加载测试集
test_set = datasets.CIFAR10(
    root='D:/pytorch/data',
    train=False,         # 这是测试集
    download=False,
    transform=transform
)

#length
train_set_size=len(train_set)
test_set_size=len(test_set)
print("训练集长度为：{}".format(train_set_size))
print("测试集长度为：{}".format(test_set_size))

#加载数据集
train_loader=DataLoader(dataset=train_set,batch_size=64)
test_loader=DataLoader(dataset=test_set,batch_size=64)

#搭建神经网络
class MLP(nn.Module):
    def __init__(self):
            super(MLP,self).__init__()
            self.model=nn.Sequential(
              nn.Linear(3072,64),
              nn.ReLU(),
              nn.Linear(64,64),
              nn.ReLU(),
              nn.Linear(64,10)
           )

    def forward(self, x):
        x=x.view(x.size(0),-1)
        return self.model(x)
import torch.optim as optim
model=MLP()
#定义损失函数和优化器
criterion=nn.CrossEntropyLoss()
optimizer=optim.Adam(model.parameters(),lr=0.001)
#训练循环
for epoch in range(10):
    running_loss=0.0
    for inputs,labels in train_loader:
        #前向传播
        outputs=model(inputs)
        loss=criterion(outputs,labels)

        #反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss+=loss.item()
    avg_loss=running_loss/len(train_loader)
    print(f'Epoch[{epoch+1}/10],Loss:{avg_loss:.4f}')

    classes=('airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck')
    def visualize_predictions(model,test_loader,num_imagines=10):
        model.eval()
        data_iter=iter(test_loader)
        images,labels=next(data_iter)
        with torch.no_grad():
            outputs=model(images)
            _,predicted=torch.max(outputs,1)
            plt.figure(figsize=(12,6))
            for i in range(num_imagines):
                img=images[i]/2+0.5
                npimg = img.numpy()
                plt.subplot(1,num_imagines,i+1)
                plt.imshow(np.transpose(npimg,(1,2,0)))
                plt.text(0,-10,f'真实标签:{classes[labels[i]]}',color='green',fontsize=12)
                plt.text(0, -30,f'预测标签:{classes[predicted[i]]}', color='red', fontsize=12)
                plt.axis('off')

        plt.show(block=True)
visualize_predictions(model,test_loader,num_imagines=10)



