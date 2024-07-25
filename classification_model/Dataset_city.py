import os
import torch
import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms


# Define data transforms for training data
transform_train = transforms.Compose([
    transforms.Resize((256, 256)),  # 统一缩放到256x256
    # transforms.RandomCrop((500, 224)),  # 随机裁剪到224x224
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


class Dataset(data.Dataset):
    def __init__(self, mode, dir):
        self.data_size = 0  # 数据集的大小
        self.img_list = []  # 用于存图
        self.img_label = []  # 标签
        self.trans = transform_train  # 转换的属性设置
        self.mode = mode  # 下面打开集的模式
        # # todo: change to your data file
        # self.img_dir = '/Classification/Dataset\\images\\'

        if self.mode == 'train':
            train_dir = dir + '/train.txt'  # 更新地址
            # Open the train.txt file
            with open(train_dir, 'r') as f:
                lines = f.readlines()  # 读取所有行

            # Split the lines by whitespace to get the file name and label
            train_data_dir = []
            for line in lines:
                filename, label = line.strip().split(',')  # 去除末尾的换行符，并按空格分割
                train_data_dir.append((filename, int(label)))  # 将文件名和标签存储为一个元组，添加到data列表中

            for file in train_data_dir:  # 遍历
                train_file_name = str(file[0])
                self.img_list.append(train_file_name)  # 存图
                self.data_size += 1
                label_x = file[1]
                self.img_label.append(label_x)  # 设置入相对应的标签
        
        elif self.mode == 'val':
            val_dir = dir + '/val.txt'  # 更新地址
            # Open the val.txt file
            with open(val_dir, 'r') as f:
                lines = f.readlines()  # 读取所有行

            # Split the lines by whitespace to get the file name and label
            val_data_dir = []
            for line in lines:
                filename, label = line.strip().split(',')  # 去除末尾的换行符，并按空格分割
                val_data_dir.append((filename, int(label)))  # 将文件名和标签存储为一个元组，添加到data列表中

            for file in val_data_dir:  # 遍历
                val_file_name = str(file[0])
                self.img_list.append(val_file_name)  # 存图
                self.data_size += 1
                label_x = file[1]
                self.img_label.append(label_x)  # 设置入相对应的标签

        elif self.mode == 'test':
            test_dir = dir + '/test.txt'
            # Open the val.txt file
            with open(test_dir, 'r') as f:
                lines = f.readlines()  # 读取所有行

            # Split the lines by whitespace to get the file name and label
            test_data_dir = []
            for line in lines:
                filename, label = line.strip().split(',')  # 去除末尾的换行符，并按空格分割
                test_data_dir.append((filename, int(label)))  # 将文件名和标签存储为一个元组，添加到data列表中

            for file in test_data_dir:  # 遍历
                test_file_name = str(file[0])
                self.img_list.append(test_file_name)  # 存图
                self.data_size += 1
                label_x = file[1]
                self.img_label.append(label_x)  # 设置入相对应的标签
        else:
            print("没有这个mode")

    def __getitem__(self, item):  # 获取数据
        if self.mode == 'train':
            img = Image.open(self.img_list[item])
            label_y = self.img_label[item]
            return self.trans(img), torch.LongTensor([label_y])  # 返回该图片的地址和标签
        elif self.mode == 'val':
            img = Image.open(self.img_list[item])
            label_y = self.img_label[item]
            return self.trans(img), torch.LongTensor([label_y])
        elif self.mode == 'test':
            img = Image.open(self.img_list[item])
            label_y = self.img_label[item]
            return self.trans(img), torch.LongTensor([label_y])
        else:
            print("None")

    def __len__(self):
        return self.data_size
