import torch
import torch.nn as nn

#定义一个继承自nn.Module的AleNet类，用于构建AlexNet模型
class AlexNet(nn.Module):
    #初始化方法，设置输入图像的形状(input_shape)和输出类别数(output_shape)
    def __init__(self, input_shape=(3, 224, 224), output_shape=2):
        #调用父类的初始化方法
        super(AlexNet, self).__init__()

        #定义特征提取部分，使用Sequential容器将多个层按顺序组合起来
        self.features = nn.Sequential(
            #第一个卷积块，包含卷积层、ReLU激活函数、批归一化层和最大池化层
            nn.Conv2d(input_shape[0], 48, kernel_size=11, stride=4, padding=0),
            nn.ReLU(inplace=True),  #ReLU激活函数,inplace=True表示直接在原地进行操作，节省内存
            nn.BatchNorm2d(48),     #批归一化层，作用域48个通道上
            nn.MaxPool2d(kernel_size=3, stride=2),      #最大池化层，池化窗口大小为3x3，步长为2

            #第二个卷积块，结构同第一个卷积块，但参数不同
            nn.Conv2d(48, 128, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),  # ReLU激活函数,inplace=True表示直接在原地进行操作，节省内存
            nn.BatchNorm2d(128),  # 批归一化层，作用域48个通道上
            nn.MaxPool2d(kernel_size=3, stride=2),  # 最大池化层，池化窗口大小为3x3，步长为2

            # 第三个卷积块，值包含卷积层和ReLU激活函数
            nn.Conv2d(128, 192, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),

            # 第四个卷积块，结构同第三个卷积块
            nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),

            # 第五个卷积块，包含卷积层、ReLU激活函数和最大池化层
            nn.Conv2d(192, 128, kernel_size=1, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )

        #计算全连接层的输入特征数量，即经过所有卷积核池化层后的特征图大小
        self.feature_size = self._get_feature_size(input_shape)

        #定义分类器部分，包括两个全连接层和一个输出层
        self.classifier = nn.Sequential(
            #第一个全连接层
            nn.Flatten(),   #将多维特征展平成一维向量
            nn.Linear(self.feature_size, 1024),     #全连接层，输入特征数为feature_size,输出特征数为1024
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.25),     # DropOut层，随机丢弃25%的神经元以防止过拟合

            # 第二个全连接层
            nn.Linear(1024, 1024),  # 全连接层，输入特征数为1024,输出特征数为1024
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.25),  # DropOut层，随机丢弃25%的神经元以防止过拟合

            #输出层，用于预测类别
            nn.Linear(1024, output_shape),  # 全连接层，输入特征数为1024,输出特征数等于类别数output_shape
        )

        #定义softmax层，用于将输出转换为概率分布
        self.softmax = nn.Softmax(dim=1)

    #内部方法，用于计算全连接层的输入特征数
    def _get_feature_size(self, input_shape):
        #创建一个虚拟输入变量，其尺寸为1加上输入形状，模拟一批次只有一个样本的情况
        x = torch.zeros(1, *input_shape)
        #将虚拟输入通过特征提取部分，获取经过所有卷积核池化层处理后的输出尺寸
        x = self.features(x)
        #返回每个样本的元素总数，即全连接层需要输入特征数
        return x.numel() // x.size(0)

    #前向传播方法，定义数据如何通过网络
    def forward(self, x):
        #首先通过特征提取部分
        x = self.features(x)
        #然后通过分类器部分
        x = self.classifier(x)
        #最后通过softmax层，得到最终输出
        x = self.softmax(x)
        #返回最后前向传播的结果
        return x

if __name__=='__main__':
    #当脚本被直接执行时(而不是被导入)运行操作
    #创建一个AlexNet模型实例
    model = AlexNet()
    #打印模型结构，以便查看模型的每一层及其参数
    print(model)
    #创建一个随机输入张量,模拟一个批次的数量，尺寸为(1, 3, 224, 224)
    x = torch.randn(1, 3, 224, 224)
    #对随机输入进行前向传播，并打印输出的形状，验证模型是否正确构建
    output = model(x)
    print(f"Output shape: {output.shape}")