import os

import os
import shutil
import random


source_img_dir = r"D:\myproject\data\pennfudanped\PNGImages"        # 原始图片文件夹
source_label_dir = r"D:\myproject\data\yolo_labels"     # YOLO标签文件夹
target_dir = r"D:\myproject\data\dataset"          # 目标文件夹
train_ratio = 0.8                              # 训练集比例

# 创建训练组，测试组
os.makedirs(os.path.join(target_dir, "images", "train"), exist_ok=True)
os.makedirs(os.path.join(target_dir, "images", "val"), exist_ok=True)
os.makedirs(os.path.join(target_dir, "labels", "train"), exist_ok=True)
os.makedirs(os.path.join(target_dir, "labels", "val"), exist_ok=True)

all_images = [f for f in os.listdir(source_img_dir) if f.endswith('.png')]
random.shuffle(all_images)  # 随机打乱，确保随机性

# ===== 计算训练集数量 =====
train_count = int(len(all_images) * train_ratio)
train_images = all_images[:train_count]
val_images = all_images[train_count:]

print("正在复制训练集...")
for img_file in train_images:
    # 复制图片
    shutil.copy(
        os.path.join(source_img_dir, img_file),
        os.path.join(target_dir, "images", "train", img_file)
    )
    # 复制对应的标签文件（把.png换成.txt）
    label_file = img_file.replace('.png', '.txt')
    shutil.copy(
        os.path.join(source_label_dir, label_file),
        os.path.join(target_dir, "labels", "train", label_file)
    )

print("正在复制验证集...")
for img_file in val_images:
    shutil.copy(
        os.path.join(source_img_dir, img_file),
        os.path.join(target_dir, "images", "val", img_file)
    )
    label_file = img_file.replace('.png', '.txt')
    shutil.copy(
        os.path.join(source_label_dir, label_file),
        os.path.join(target_dir, "labels", "val", label_file)
    )

print(f"完成！训练集：{len(train_images)}张，验证集：{len(val_images)}张")