# 导入必要的库和模块
import sys  # 系统相关参数和函数
import torch  # PyTorch，用于深度学习模型的构建与训练
import cv2  # OpenCV，用于图像处理任务
import numpy as np  # NumPy，用于数值计算
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QFileDialog, QVBoxLayout, QWidget  # PyQt5窗口部件
from PyQt5.QtGui import QImage, QPixmap  # PyQt5图像处理类
from PyQt5.QtCore import Qt  # PyQt5核心功能，如对齐方式等
from torchvision import transforms  # 包含图像预处理方法的模块
from model.AlexNet import AlexNet  # 自定义的AlexNet模型
import utils  # 自定义工具模块，包含辅助函数


class ImageClassifierApp(QMainWindow):
    def __init__(self):
        super().__init__()  # 调用父类构造函数
        self.initUI()  # 初始化用户界面
        self.setupModel()  # 设置模型

    def initUI(self):
        # 设置窗口的基本属性
        self.setWindowTitle('图像分类器')  # 设置窗口标题
        self.setGeometry(100, 100, 800, 600)  # 设置窗口位置和大小

        # 创建中心部件和布局管理器
        central_widget = QWidget()  # 创建一个QWidget作为主窗口的中心部件
        self.setCentralWidget(central_widget)  # 将创建的部件设置为中心部件
        layout = QVBoxLayout(central_widget)  # 创建垂直布局管理器，并将其应用于中心部件

        # 创建图像显示标签
        self.image_label = QLabel()  # 创建一个QLabel用于显示图像
        self.image_label.setAlignment(Qt.AlignCenter)  # 设置图像居中对齐
        self.image_label.setMinimumSize(400, 400)  # 设置最小尺寸以确保有足够的空间显示图像
        layout.addWidget(self.image_label)  # 将图像标签添加到布局中

        # 创建结果显示标签
        self.result_label = QLabel('预测结果将在这里显示')  # 创建一个QLabel用于显示预测结果
        self.result_label.setAlignment(Qt.AlignCenter)  # 设置文本居中对齐
        layout.addWidget(self.result_label)  # 将结果标签添加到布局中

        # 创建按钮
        self.select_button = QPushButton('选择图片')  # 创建“选择图片”按钮
        self.select_button.clicked.connect(self.selectImage)  # 连接按钮点击事件到selectImage方法
        layout.addWidget(self.select_button)  # 将按钮添加到布局中

        self.predict_button = QPushButton('开始预测')  # 创建“开始预测”按钮
        self.predict_button.clicked.connect(self.predict)  # 连接按钮点击事件到predict方法
        self.predict_button.setEnabled(False)  # 初始时禁用预测按钮
        layout.addWidget(self.predict_button)  # 将按钮添加到布局中

        # 初始化图像变量
        self.current_image = None  # 当前加载的图像初始化为None

    def setupModel(self):
        # 设置设备（GPU或CPU）
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 根据可用性选择CUDA或CPU

        try:
            # 加载模型
            self.model = AlexNet()  # 实例化AlexNet模型
            self.model = self.model.float()  # 确保模型是float32类型
            self.model.load_state_dict(torch.load("./logs/final_model.pth", map_location=self.device))  # 加载预训练权重
            self.model = self.model.to(self.device)  # 将模型移动到指定设备上
            self.model.eval()  # 将模型设置为评估模式

        except Exception as e:
            print(f"Model setup error: {str(e)}")  # 如果发生错误，打印错误信息
            raise  # 抛出异常

    def selectImage(self):
        # 打开文件对话框选择图片
        file_name, _ = QFileDialog.getOpenFileName(
            self, "选择图片", "", "图像文件 (*.jpg *.jpeg *.png *.bmp)")  # 弹出文件选择对话框

        if file_name:  # 如果选择了文件
            try:
                # 读取并显示图片
                self.current_image = cv2.imread(file_name)  # 使用OpenCV读取图像文件
                if self.current_image is not None:  # 如果图像成功加载
                    # 调整图像大小以适应显示
                    display_image = self.resizeImage(self.current_image.copy(), 400)  # 调用resizeImage方法调整图像大小
                    # 转换为Qt图像格式并显示
                    height, width, channel = display_image.shape  # 获取图像的高度、宽度和通道数
                    bytes_per_line = 3 * width  # 计算每行字节数
                    qt_image = QImage(
                        display_image.data, width, height, bytes_per_line,
                        QImage.Format_RGB888).rgbSwapped()  # 将图像转换为QImage格式
                    self.image_label.setPixmap(QPixmap.fromImage(qt_image))  # 将QImage设置为标签的图像
                    # 启用预测按钮
                    self.predict_button.setEnabled(True)  # 允许点击预测按钮
                    # 清除之前的预测结果
                    self.result_label.setText('图片已加载，点击"开始预测"进行预测')  # 更新结果标签文本
                else:
                    self.result_label.setText('图片加载失败')  # 如果图像加载失败，更新结果标签文本
            except Exception as e:
                print(f"Image loading error: {str(e)}")  # 如果发生错误，打印错误信息
                self.result_label.setText('图片加载出错')  # 更新结果标签文本

    def resizeImage(self, image, target_size):
        # 调整图像大小，保持纵横比
        h, w = image.shape[:2]  # 获取图像的高度和宽度
        if h > w:  # 如果高度大于宽度
            new_h = target_size  # 新的高度为目标尺寸
            new_w = int(w * target_size / h)  # 新的宽度按比例缩放
        else:
            new_w = target_size  # 新的宽度为目标尺寸
            new_h = int(h * target_size / w)  # 新的高度按比例缩放
        return cv2.resize(image, (new_w, new_h))  # 返回调整后的图像

    def predict(self):
        if self.current_image is not None:  # 如果有当前图像
            try:
                # 1. 首先裁剪为中心正方形
                h, w = self.current_image.shape[:2]  # 获取图像的高度和宽度
                size = min(h, w)  # 获取较短的一边作为裁剪尺寸
                y_start = (h - size) // 2  # 计算y轴起始位置
                x_start = (w - size) // 2  # 计算x轴起始位置
                cropped_img = self.current_image[y_start:y_start + size, x_start:x_start + size]  # 裁剪图像

                # 2. 转换颜色空间并归一化
                img_RGB = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)  # 将BGR格式转换为RGB格式
                img_resized = cv2.resize(img_RGB, (224, 224))  # 调整图像大小到224x224
                img_float = img_resized.astype(np.float32) / 255.0  # 归一化像素值到[0, 1]

                # 3. 转换为 tensor 并调整维度
                img_tensor = torch.from_numpy(img_float).permute(2, 0, 1)  # 将numpy数组转换为torch张量，并调整维度顺序
                img_tensor = img_tensor.unsqueeze(0)  # 增加一个批次维度

                # 4. 标准化
                normalize = transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],  # 定义标准化的均值
                    std=[0.229, 0.224, 0.225]  # 定义标准化的标准差
                )
                img_tensor = normalize(img_tensor)  # 对张量应用标准化

                # 5. 移动到设备
                img_tensor = img_tensor.to(self.device, dtype=torch.float32)  # 将张量移动到指定设备，并确保数据类型为float32

                # 6. 预测
                self.model.eval()  # 确保模型在评估模式
                with torch.no_grad():  # 关闭梯度计算以节省内存
                    outputs = self.model(img_tensor)  # 模型推理，获取输出
                    # 注意：模型已经包含了 softmax 层，不需要再次应用
                    confidence, predicted = torch.max(outputs, 1)  # 获取最大值及其索引（即预测类别）

                    # 获取预测结果和置信度
                    result = utils.print_answer(predicted.item())  # 调用utils中的print_answer函数获取类别名称
                    confidence_value = confidence.item() * 100  # 获取置信度并转换为百分比

                    # 显示预测结果和置信度
                    self.result_label.setText(
                        f'预测结果: {result}\n置信度: {confidence_value:.2f}%')  # 更新结果标签文本

            except Exception as e:
                print(f"Prediction error: {str(e)}")  # 如果发生错误，打印错误信息
                self.result_label.setText(f'预测出错: {str(e)}')  # 更新结果标签文本
        else:
            self.result_label.setText('请先选择一张图片')  # 如果没有选择图片，提示用户选择图片


def main():
    app = QApplication(sys.argv)  # 创建应用程序对象
    window = ImageClassifierApp()  # 创建主窗口实例
    window.show()  # 显示窗口
    sys.exit(app.exec_())  # 进入应用程序的主循环，并在退出时清理资源


if __name__ == '__main__':
    main()  # 如果脚本被直接运行，则调用main函数启动应用程序