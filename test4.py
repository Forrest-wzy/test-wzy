import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torchvision
from fontTools.misc.iterTools import batched
from sklearn.model_selection import train_test_split
from torch.cuda import device
from torch.utils.data import DataLoader
#torchvision
from torchvision import datasets
from torchvision import transforms
transform = transforms.Compose([
    transforms.ToTensor(),  # 把图片从PIL格式转成PyTorch张量
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 归一化
])

# 2. 下载/加载训练集
train_set = datasets.CIFAR10(
    root='D:\myproject\data',       # 存到当前目录的 data 文件夹
    train=True,          # 这是训练集
    download=False,
    transform=transform  # 应用预处理
)

# 3. 下载/加载测试集
test_set = datasets.CIFAR10(
    root='D:\myproject\data',
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


#残差连接
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)  # 旁路
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity  # 残差连接
        out = self.relu(out)
        return out


class ImprovedCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            # 第一组：3→16，带残差
            ResidualBlock(3, 16),
            nn.MaxPool2d(2),

            # 第二组：16→32，带残差
            ResidualBlock(16, 32),
            nn.MaxPool2d(2),

            # 第三组：32→64，带残差
            ResidualBlock(32, 64),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

device=torch.device('cuda'if torch.cuda.is_available()else'cpu')
print(f"Using device:{device}")

model = ImprovedCNN().to(device)
#损失函数
criterion=nn.CrossEntropyLoss()
#优化器
learning_rate=0.001
optimizer=optim.Adam(model.parameters(),lr=learning_rate)

#训练模型
print("begin CNN")
train_losses=[]
for epoch in range(10):
    running_loss=0.0
    for i,(images,labels) in enumerate(train_loader):
        images,labels=images.to(device),labels.to(device)


        outputs=model(images)
        loss = criterion(outputs, labels)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    avg_loss = running_loss / len(train_loader)
    train_losses.append(avg_loss)
    print(f'Epoch[{epoch+1}/10],Loss:{avg_loss:.4f}')

#测试
correct=0
total=0
with torch.no_grad():
    for images,labels in test_loader:
        images,labels=images.to(device),labels.to(device)
        outputs=model(images)
        _,predicted=torch.max(outputs,1)
        total +=labels.size(0)
        correct +=(predicted==labels).sum().item()
print(f'\nCNN准确率：{100*correct/total:.2f}%')

#损失曲线
plt.plot(train_losses)
plt.title('CNN training loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.savefig('CNN_loss.png')
plt.show(block=True)


#warmup
from torch.optim.lr_scheduler import LinearLR,SequentialLR
optimizer=optim.Adam(model.parameters(),lr=0.001)
scheduler=LinearLR(
    optimizer,
    start_factor=0.01,
    end_factor=1.0,
    total_iters=5*len(train_loader),
    last_epoch=-1
)

def mixup_data(x,y,alpha=1.0):
    if alpha>0:
        lam=np.random.beta(alpha,alpha)
    else:
        lam=1
    batch_size=x.size(0)
    index=torch.randperm(batch_size).to(x.device)
    mixed_x=lam*x+(1-lam)*x[index]
    y_a,y_b=y,y[index]
    return mixed_x,y_a,y_b,lam
#训练
print("begin")

train_losses=[]
for epoch in range(10):
    model.train()
    running_loss=0.0
    print(f"当前学习率：{optimizer.param_groups[0]['lr']}")

    for images,labels in train_loader:
        images,labels=images.to(device),labels.to(device)
        images,labels_a,labels_b,lam=mixup_data(images,labels,alpha=1.0)
        outputs=model(images)
        loss=lam*criterion(outputs,labels_a)+(1-lam)*criterion(outputs,labels_b)
        loss.backward()

        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        running_loss += loss.item()

avg_loss = running_loss / len(train_loader)
train_losses.append(avg_loss)
print(f'Epoch{epoch+1},Loss:{avg_loss:.4f}')

#测试准确率
model.eval()
correct=0
total=0
with torch.no_grad():
    for images,labels in test_loader:
        images,labels=images.to(device),labels.to(device)
        outputs=model(images)
        _,predicted=torch.max(outputs,1)
        total +=labels.size(0)
        correct +=(predicted==labels).sum().item()

accuracy=100*correct/total
print(f"准确率:{accuracy:.2f}%")
#记录结果
with open("test4_result.txt","a")as f:
    f.write(f"warmup(5)+mixup(alpha=1.0),准确率：{accuracy:.2f}%\n")

# 计算参数量和计算量
from thop import profile, clever_format

model.eval()
dummy_input = torch.randn(1, 3, 32, 32).to(device)
flops, params = profile(model, inputs=(dummy_input,), verbose=False)
flops, params = clever_format([flops, params], "%.3f")

print("\n" + "=" * 50)
print("模型轻量化指标")
print("=" * 50)
print(f"参数量 (Params): {params}")
print(f"计算量 (FLOPs): {flops}")
print("=" * 50)

with open("task4_results.txt", "a") as f:
    f.write(f"参数量: {params}, 计算量: {flops}\n")