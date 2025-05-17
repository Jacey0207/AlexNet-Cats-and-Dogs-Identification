# 导入os模块，用于与操作系统进行交互，例如文件和目录操作
import os

# 使用os.listdir函数列出指定路径下的所有文件名，并将结果赋值给photos变量
# 路径为"./data/image/train/"，表示当前目录下data文件夹中的image子文件夹中的train子文件夹
photos = os.listdir("./data/image/train/")

# 以下代码块用于创建或打开一个名为"data/dataset.txt"的文件，以写入模式（"w"）操作该文件
# 如果文件已经存在，则会清空文件内容；如果文件不存在，则会创建新文件
with open("data/dataset.txt", "w") as f:
    # 遍历photos列表中的每一个文件名
    for photo in photos:
        # 将文件名按照"."分割，取第一个部分作为name变量的值，即去掉文件扩展名
        name = photo.split(".")[0]
        # 检查name是否等于"cat"
        if name == "cat":
            # 如果是猫的图片，则向文件中写入图片文件名后跟";0"和换行符，表示类别标签为0
            f.write(photo + ";0\n")
        # 检查name是否等于"dog"
        elif name == "dog":
            # 如果是狗的图片，则向文件中写入图片文件名后跟";1"和换行符，表示类别标签为1
            f.write(photo + ";1\n")

# 关闭文件。注意：由于使用了with语句，这里实际上不需要显式调用f.close()，因为with语句会在结束时自动关闭文件
f.close()