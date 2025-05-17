# 导入matplotlib.image模块用于读取图像文件
import matplotlib.image as mpimg
# 导入numpy模块，通常用于处理数值数据和数组操作
import numpy as np
# 导入OpenCV库，主要用于计算机视觉任务，如图像处理等
import cv2
# 导入PyTorch库，用于深度学习模型的构建与训练
import torch
# 从torchvision中导入transforms，它提供了一系列图像预处理方法
from torchvision import transforms
# 导入PIL中的Image模块，用于创建、加载和处理图像
from PIL import Image


def load_image(path):
    """读取并裁剪图片为中心正方形"""
    # 使用mpimg.imread函数根据给定路径读取图像文件，并将图像存储为RGB格式的NumPy数组
    img = mpimg.imread(path)
    # 计算图像较短的一边的长度，用于确定要裁剪的正方形区域的大小
    short_edge = min(img.shape[:2])
    # 计算图像在y轴方向上需要裁剪掉的顶部和底部像素数量，以获得中心正方形
    yy = int((img.shape[0] - short_edge) / 2)
    # 计算图像在x轴方向上需要裁剪掉的左侧和右侧像素数量，以获得中心正方形
    xx = int((img.shape[1] - short_edge) / 2)
    # 根据计算出的位置参数裁剪图像，得到中心正方形区域
    crop_img = img[yy: yy + short_edge, xx: xx + short_edge]
    # 返回裁剪后的图像
    return crop_img


def resize_image(image, size):
    """调整图片大小"""
    images = []
    # 检查输入的图像是否是三维数组（即单张图像），如果是，则添加一个维度以模拟批次（batch）
    if len(image.shape) == 3:
        image = np.expand_dims(image, axis=0)

    # 遍历每个图像（即使是单张图像，此时也被视为一个批次的一部分）
    for i in image:
        if isinstance(i, np.ndarray):
            # 如果当前图像是NumPy数组，则将其转换为PIL图像对象。这里乘以255并将数据类型转换为uint8，因为PIL期望的是0-255范围内的整数像素值
            i = Image.fromarray((i * 255).astype(np.uint8))
        # 定义一系列图像预处理步骤：先调整大小至指定尺寸，然后转换为PyTorch张量，这一步会自动将像素值归一化到[0,1]之间
        transform = transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor()  # 将图像转换为PyTorch张量，并归一化到[0,1]
        ])
        # 应用定义的转换到图像上
        i = transform(i)
        # 将处理后的图像添加到列表中
        images.append(i)

    # 使用torch.stack将所有图像堆叠成一个张量，表示整个批次
    images = torch.stack(images)
    # 返回调整大小后的图像批次
    return images


def print_answer(argmax):
    """打印预测结果"""
    # 打开包含类别索引到名称映射的文本文件
    with open("./data/model/index_word.txt", "r", encoding='utf-8') as f:
        # 读取文件中的每一行，并去除每行末尾的换行符后只保留类别名称部分，创建一个类别名称列表
        synset = [l.split(";")[1][:-1] for l in f.readlines()]

    # 检查argmax是否是PyTorch张量，如果是，则调用item()方法获取其Python标量值
    if torch.is_tensor(argmax):
        argmax = argmax.item()

    # 打印对应于最大值索引的类别名称
    print(synset[argmax])
    # 返回类别名称，以便可以在其他地方使用这个函数的返回值
    return synset[argmax]