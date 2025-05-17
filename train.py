import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader    #用于创建自定义数据集和加载器
import numpy as np
import cv2  # OpenCV,用于文件路径操作等
from torchvision import transforms      #包含图像处理方法的模块
from torch.optim import SGD     #随机梯度下降的优化器
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts    #学习率调度器
import matplotlib.pyplot as plt     #绘制图表
from model.AlexNet import AlexNet

#定义训练数据增强策略，包括随即裁剪、水平翻转、旋转以及颜色抖动等操作，并进行归一化处理
train_transform = transforms.Compose([
    transforms.ToPILImage(),    #将Numpy数组或张量转换为PIL图像
    transforms.RandomResizedCrop(224),      #随机大小裁剪并调整到224x224大小
    transforms.RandomHorizontalFlip(p=0.5),     #随机水平翻转
    transforms.RandomRotation(15),      #随机旋转最多15度
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),   #随即改变亮度，对比度和饱和度
    transforms.ToTensor(),      #转换为Pytorch张量并归一化到[0,1]
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])     #使用给定的均值和标准差对每个通道进行标准化
])

#定义验证集转换，只包含基本的尺寸调整和中心裁剪，并进行归一化处理
val_transform = transforms.Compose([
    transforms.ToPILImage(),        #将Numpy数组或张量转换为PIL图像
    transforms.Resize(256),     #将最短边调整到256像素
    transforms.CenterCrop(224),     #中心裁剪到224x224大小
    transforms.ToTensor(),      #转换为PyTorch张量并归一化到[0,1]
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])     #使用给定的均值和标准差对每个通道进行标准化
])

class CustomDataset(Dataset):
    #初始化并开启训练模式
    def __init__(self, lines, transform=None, is_train=True):
        self.lines = lines  #数据行列表，每行对应一个样本的信息
        self.transform = transform      #应用的数据转换
        self.is_train = is_train        #标记是否为训练集

    def __len__(self):
        return len(self.lines)      #返回数据中样本的数量

    def __getitem__(self, idx):
        try:
            #解析每一行的数据，获取图像文件名和标签
            name = self.lines[idx].strip().split(';')[0]
            img = cv2.imread(os.path.join("./data/image/train", name))  #使用OpenCV获取图像
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  #将BGR格式转换为RGB格式
            label = int(self.lines[idx].strip().split(';')[1])  #获取图像对应的标签

            if self.transform:
                img = self.transform(img)       #对图像应用指定的转换

            return img, label   #返回处理后的图像和标签
        except Exception as e:
            print(f"Error loading image {name}: {str(e)}")  #如果读取过程中出现问题，打印错误信息
            return self.__getitem__(0)      #返回第一个样本作为替代

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs, log_dir, patience=7):
    history = {
        'train_loss': [], 'train_acc': [],      #记录训练损失和准确率
        'val_loss': [], 'val_acc': [],      #记录验证损失和准确率
        'lr': []    #记录学习率变化
    }

    best_val_acc = 0    #初始化最佳验证集的准确率为0
    counter = 0     #早停计数器初始化为0

    for epoch in range(num_epochs):     #开始训练循环，迭代num_epochs
        #训练阶段
        model.train()       #设置模型为训练模式
        train_loss = 0      #初始化训练损失为0
        train_correct = 0   #初始化训练预测正确数为0
        train_total = 0     #初始化训练样本总数为0

        for batch_odx, (images, labels) in enumerate(train_loader):      #遍历训练数据加载器
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()   #清空之前计算的梯度
            outputs = model(images)     #前向传播，得到模型输出
            loss = criterion(outputs, labels)       #计算损失
            loss.backward()     #反向传播

            #梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()    #更新模型参数

            train_loss += loss.item()   #累加损失
            _, predicted = outputs.max(1)   #获取预测结果
            train_total += labels.size(0)   #累加样本总数
            train_correct += predicted.eq(labels).sum().item()      #累加正确预测数

        train_loss = train_loss / len(train_loader)     #计算平均训练损失
        train_acc = 100. * train_correct / train_total      #计算训练准确率

        #验证阶段
        model.eval()    #设置模型为评估模式
        val_loss = 0       #验证损失
        val_correct = 0     #验证正确预测数
        val_total = 0

        with torch.no_grad():       #关闭梯度计算，减少内存消耗
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()     #累加损失
                _, predicted = outputs.max(1)   #获取预测结果
                val_total += labels.size(0)     #累加样本总数
                val_correct += predicted.eq(labels).sum().item()    #累加正确预测数

        val_loss = val_loss / len(val_loader)   #计算平均验证损失
        val_acc = 100. * val_correct / val_total     #计算验证集的准确率

        #更新学习率
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]     #获取当前的学习率

        #更新历史纪录
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['lr'].append(current_lr)

        #保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            counter = 0     #重置早停计数器
            torch.save(model.state_dict(), os.path.join(log_dir, 'best_model.pth'))
        else:
            counter += 1    #否则增加早停计数器

        #每三个epoch保存一次
        if (epoch + 1) % 3 == 0:
            torch.save(model.state_dict(), os.path.join(log_dir, f'ep{epoch:03d}-loss{train_loss:.3f}-val_loss{val_loss:.3f}.pth'))     #保存模型状态
        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, '
              f'Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')

        if counter >= patience:     #如果早停计数器达到设定的耐心值
            print(f"Early stopping triggered after {epoch + 1} epochs")     #触发早停机制
            break   #结束训练循环

    return history      #返回训练历史记录

#可视化
def plot_training_history(history, log_dir):
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)    #在1行3列的网格中创建第1个子图
    plt.plot(history['train_loss'], label='Train Loss')     #绘制训练损失曲线
    plt.plot(history['val_loss'], label='Val Loss')     #绘制验证损失曲线
    plt.title("Loss History")
    plt.xlabel("Epoch")
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(history['train_acc'], label='Train Acc')     #绘制训练准确率曲线
    plt.plot(history['val_acc'], label='Val Acc')     #绘制验证准确率曲线
    plt.title("Accuracy History")
    plt.xlabel("Epoch")
    plt.ylabel('Accuracy (%)')
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(history['lr'])     #绘制学习率变化曲线
    plt.title("Learning Rate History")
    plt.xlabel("Epoch")
    plt.ylabel('Learning Rate')
    plt.legend()

def main():
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    np.random.seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    #创建日志记录，如果不存在则创建
    log_dir = "./logs/"
    os.makedirs(log_dir, exist_ok=True)

    #加载数据，从文本文件中读取所有数据行
    with open("./data/dataset.txt", "r") as f:
        lines = f.readlines()

    #打乱数据顺序
    np.random.shuffle(lines)

    #划分数据集,10%用于验证
    num_val = int(len(lines) * 0.1)
    train_lines = lines[:-num_val]
    val_lines = lines[-num_val:]

    #创建数据实例，分别针对训练集和验证集
    train_dataset = CustomDataset(train_lines, transform=train_transform, is_train=True)    #测试模型要打开
    val_dataset = CustomDataset(val_lines, transform=val_transform, is_train=False)     #验证集不需要开测试模式（只要是取标签的）

    #创建数据加载器，用于批量加载数据
    batch_size = 64
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    #初始化模型，并将其移动到选定的设备上
    model = AlexNet().to(device)

    #损失函数和优化器初始化
    criterion = nn.CrossEntropyLoss()   #使用交叉熵损失函数
    optimizer = SGD(
        model.parameters(),     #传递模型参数给优化器
        lr=0.01,
        momentum=0.9,       #动量因子
        weight_decay=5e-4,      #权重衰减（L2正则化）
        nesterov=True       #使用Nesterov加速梯度
    )

    #学习率调度器初始化
    scheduler = CosineAnnealingWarmRestarts(
        optimizer,      #传递优化器给调度器
        T_0=10,     #第一个周期的长度
        T_mult=2,   #每个新周期的长度事前一个周期的两倍
        eta_min=1e-6   #最小学习率
    )

    #设置训练参数
    num_epochs = 50
    patience = 7    #早停机制的耐心值

    print(f'Training on {len(train_lines)} samples, validating on {len(val_lines)} samples')    #打印训练和验证样本数
    print(f'Batch size: {batch_size}')      #打印批次大小

    #训练模型
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_epochs=num_epochs,
        log_dir=log_dir,
        patience=patience
    )

    #绘制训练历史
    plot_training_history(history, log_dir)

    #保存最终模型
    torch.save(model.state_dict(), os.path.join(log_dir, 'final_model.pth'))

if __name__=='__main__':
    #如果脚本被直接运行，则调用main()函数启动训练过程
    main()