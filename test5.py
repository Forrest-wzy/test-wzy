import os
import re
from tkinter import Image
from PIL import Image
import numpy as np
import torch
from torch import nn
import torch.optim as optim
import torch.nn as nn
from torch.cuda import device
from torch.utils.data import Dataset,DataLoader
import matplotlib.pyplot as plt
from torchvision import  transforms

#准备
class unetdataset(Dataset):
    def __init__(self,root,split='train',target_size=(128,128)):
        self.root=root
        self.split=split
        self.target_size=target_size
        all_masks=sorted(os.listdir(os.path.join(root,'PedMasks')))
        if split=='train':
            self.masks=all_masks[:136]
        else:
            self.masks=all_masks[136:]
        print(f"{split}集: {len(self.masks)}张图片")


    def __len__(self):
        return len(self.masks)

    def __getitem__(self,index):
        mask_name=self.masks[index]
        img_name=mask_name.replace('_mask.png','.png')

        img=Image.open(os.path.join(self.root,'PNGImages',img_name)).convert('RGB')#(三个通道红绿蓝)
        mask=Image.open(os.path.join(self.root,'PedMasks',mask_name))
        #统一尺寸
        img = img.resize(self.target_size, Image.BILINEAR)
        mask = mask.resize(self.target_size, Image.NEAREST)

        transform=transforms.ToTensor()
        img_tensor=transform(img)
        mask_tensor=transform(mask)
        return transform(img),transform(mask)

    #测试数据集
if __name__=='__main__':
    dataset_path=r'D:\myproject\data\pennfudanped'
    train_dataset=unetdataset(dataset_path,'train')
    val_dataset=unetdataset(dataset_path,'val')

    print(f"训练集样本数：{len(train_dataset)}")
    print(f"验证集样本数：{len(val_dataset)}")

    #可视化第一个样本
    img,mask=train_dataset[0]
    plt.figure(figsize=(8,4))
    plt.subplot(1,2,1)
    plt.imshow(img.permute(1,2,0))
    plt.title('行人图片')
    plt.subplot(1,2,2)
    plt.imshow(mask.squeeze(),cmap='gray')
    plt.title('分割掩码')
    plt.show(block=True)

#搭建unet网络
import torch.nn.functional as F
class doubleConv(nn.Module):
    #两次卷积+BN+ReLU
    def __init__(self,in_channels,out_channels):
        super(doubleConv,self).__init__()
        self.double_conv=nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size=3,padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels,out_channels,kernel_size=3,padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        return self.double_conv(x)

class down(nn.Module):#下采样：最大池化+双卷积
    def __init__(self,in_channels,out_channels):
        super(down,self).__init__()
        self.maxpool_conv=nn.Sequential(
            nn.MaxPool2d(2),
            doubleConv(in_channels,out_channels)
        )

    def forward(self,x):
        return self.maxpool_conv(x)

class up(nn.Module):#上采样：转置卷积+跳跃连接+双卷积
    def __init__(self,in_channels,out_channels):
        super(up,self).__init__()
        self.up=nn.ConvTranspose2d(in_channels,in_channels//2,kernel_size=2,stride=2)#转置卷积
        self.conv=doubleConv(in_channels,out_channels)

    def forward(self,x1,x2):
        x1=self.up(x1)
        #处理尺寸不匹配（填充）
        diffY=x2.size()[2] -x1.size()[2]#计算高差
        diffX=x2.size()[3] -x1.size()[3]#计算宽差
        x1=F.pad(x1,[diffX//2,diffX-diffX//2,
                          diffY//2,diffY-diffY//2])#F.pad填充
        x=torch.cat([x2,x1],dim=1)#跳跃连接（拼接）
        return self.conv(x)

class outConv(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(outConv,self).__init__()
        self.conv=nn.Conv2d(in_channels,out_channels,kernel_size=1)

    def forward(self,x):
        return self.conv(x)

class unet(nn.Module):
    def __init__(self,n_channels=3,n_classes=1,features=[64,128,256,512]):
        super(unet,self).__init__()

        #编码器（下采样路径）
        self.inc=doubleConv(n_channels,features[0])
        self.down1=down(features[0],features[1])
        self.down2 = down(features[1], features[2])
        self.down3 = down(features[2], features[3])
        self.down4 = down(features[3], features[3] * 2)  # 最底层

        #解码器（上采样路径）
        self.up1 = up(features[3] * 2, features[3])
        self.up2 = up(features[3], features[2])
        self.up3 = up(features[2], features[1])
        self.up4 = up(features[1], features[0])

        self.outcome=outConv(features[0],n_classes)
        self.sigmoid=nn.Sigmoid()

    def forward(self,x):
        #编码器
        x1 = self.inc(x)#第一层输出（用于跳跃连接）
        x2 = self.down1(x1)#第二层输出
        x3 = self.down2(x2)#第三层输出
        x4 = self.down3(x3)#第四层输出
        x5 = self.down4(x4)#最底层

        #解码器
        x = self.up1(x5, x4)#拼接x4
        x = self.up2(x, x3)#拼接x3
        x = self.up3(x, x2)#拼接x2
        x = self.up4(x, x1)#拼接x1

        x=self.outcome(x)
        return self.sigmoid(x)#输出0-1之间的概率

if __name__=='__main__':
    model=unet(n_channels=3,n_classes=1)
    x=torch.randn(1,3,128,128)
    y=model(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {y.shape}")
    print(f"参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

#训练unet
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data_root=r'D:\myproject\data\pennfudanped'
batch_size=16
epochs=30
lr=1e-4

#加载数据
train_dataset=unetdataset(dataset_path,'train')
val_dataset=unetdataset(dataset_path,'val')
train_loader=DataLoader(train_dataset,batch_size=batch_size,shuffle=True,num_workers=0)
val_loader=DataLoader(val_dataset,batch_size=batch_size,shuffle=False,num_workers=0)
print(f"训练批次:{len(train_loader)}")
print(f"验证批次:{len(val_loader)}")

#初始化模型
model=unet(n_channels=3,n_classes=1).to(device)
criterion=nn.BCELoss()#二值交叉熵
optimizer=optim.Adam(model.parameters(),lr=lr)

#训练循环
best_val_loss=float('inf')
for epoch in range(epochs):
    model.train()
    train_loss=0
    for images,masks in train_loader:
        images,masks=images.to(device),masks.to(device)
        #前向传播
        outputs=model(images)#模型猜的掩码
        loss=criterion(outputs,masks)

        #反向传播(只有训练要，因为要学习）
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss +=loss.item()
    avg_train_loss=train_loss/len(train_loader)

    #验证
    model.eval()
    val_loss=0
    with torch.no_grad():
        for images,masks in val_loader:
            images,masks=images.to(device),masks.to(device)
            outputs = model(images)  # 模型猜的掩码
            loss = criterion(outputs, masks)
            val_loss +=loss.item()
        avg_val_loss=val_loss/len(val_loader)

    print(f'epoch [{epoch+1}/{epochs}]'
          f'train_loss:{avg_train_loss:.4f}'
          f'val_loss:{avg_val_loss:.4f}')

    if avg_val_loss < best_val_loss:
        best_val_loss=avg_val_loss
        torch.save(model.state_dict(),'best_unet_model.pth')
        print(f'最佳模型，验证损失：{best_val_loss:.4f}')

print('train finished')


def predict_unet(image_path, unet_model, device):
    img = Image.open(image_path).convert('RGB')
    transform = transforms.ToTensor()
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        mask = unet_model(img_tensor)
        mask = mask.squeeze().cpu().numpy()

    return img, mask

# 现在调用预测函数
test_image = r"D:\myproject\data\dataset\images\val\FudanPed00055.png"
img, mask = predict_unet(test_image, model, device)

plt.figure(figsize=(12,4))
plt.subplot(1,3,1)
plt.imshow(img); plt.title('原图')
plt.subplot(1,3,2)
plt.imshow(mask, cmap='gray')
plt.title('预测掩码')
plt.show()

#可视化
plt.figure(figsize=(15, 5))

# 子图1：原图
plt.subplot(1, 3, 1)
plt.imshow(img)
plt.title('原图')
plt.axis('off')

# 子图2：预测掩码（黑白）
plt.subplot(1, 3, 2)
plt.imshow(mask, cmap='gray')
plt.title('预测掩码')
plt.axis('off')

# 子图3：掩码叠加到原图
plt.subplot(1, 3, 3)
# 把PIL图片转成numpy数组
img_array = np.array(img)
# 创建叠加图（复制原图）
overlay = img_array.copy()
# 掩码中大于0.5的地方标记为红色
overlay[mask > 0.2] = [255, 0, 0]  # 红色
plt.imshow(overlay)
plt.title('叠加效果（红色为行人）')
plt.axis('off')

plt.tight_layout()
plt.show()