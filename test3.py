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


#CNN模型
class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.model=nn.Sequential(
            #3-16
            nn.Conv2d(3,16,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            #16-32
            nn.Conv2d(16,32,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            #32-64
            nn.Conv2d(32,64,kernel_size=3,padding=1),
            nn.MaxPool2d(2),
        )
        self.fl=nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*4*4,256),
            nn.ReLU(),
            nn.Linear(256,10)
        )
    def forward(self,x):
        x=self.model(x)
        print("卷积后的尺寸",x.shape)
        x=self.fl(x)
        return x

device=torch.device('cuda'if torch.cuda.is_available()else'cpu')
print(f"Using device:{device}")
model=CNN()
model=model.to(device)
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
print(f'\n准确率：{100*correct/total:.2f}%')

#损失曲线
plt.plot(train_losses)
plt.title('CNN training loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.savefig('CNN_loss.png')
plt.show(block=True)