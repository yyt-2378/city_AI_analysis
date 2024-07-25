import torch
from model import Resnet_50
from torch.utils.data import DataLoader as DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import os
from Dataset_city import Dataset
import numpy as np
import time
from tqdm import tqdm as tqdm


if __name__ == '__main__':
    # Define ResNet50 model architecture
    resnet = Resnet_50(num_classes=4)

    if torch.cuda.is_available():
        resnet = resnet.cuda()  # 使用gpu训练
    model = nn.DataParallel(resnet)  # 传入
    model.train()  # 训练模式

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # Load training and testing data
    trainset = Dataset('train', dir='D:\\project\\detect_pest\\Classification\\Dataset')
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=2, shuffle=True, num_workers=2, drop_last=True)

    valset = Dataset('val', dir='D:\\project\\detect_pest\\Classification\\Dataset')
    valloader = torch.utils.data.DataLoader(valset, batch_size=2, shuffle=True, num_workers=2, drop_last=True)

    testset = Dataset('test', dir='D:\\project\\detect_pest\\Classification\\Dataset')
    testloader = torch.utils.data.DataLoader(testset, batch_size=2, shuffle=True, num_workers=2, drop_last=True)

    # Train the model
    # 添加tensorboard
    writer = SummaryWriter("logs_train")
    model_dir = './model/'  # 网络参数保存位置
    os.makedirs(model_dir, exist_ok=True)
    start_time = time.time()  # 开始训练的时间
    total_train_step = 0  # 训练图片数量
    nepoch = 200

    for epoch in tqdm(range(nepoch)):
        train_loss = []
        train_accuracy = []
        for img, label in trainloader:
            img, label = Variable(img), Variable(label)  # 将数据放置在PyTorch的Variable节点中
            img, label = img.cuda(), label.cuda()
            out = model(img)
            loss = criterion(out, label.squeeze())  # 计算损失
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss.append(loss.cpu().item())
            predicted = torch.argmax(out, 1)
            train_correct = (predicted == label.squeeze()).sum().item()
            train_total = label.size(0)
            total_train_step += 1
            train_accuracy.append(train_correct / train_total)

            # print('Epoch:{0},Frame:{1}, train_loss {2}, train_accuracy {3}'.format(epoch, total_train_step*batch_size, loss/batch_size, train_accuracy))

            if total_train_step % 40 == 0:
                end_time = time.time()  # 训练结束时间
                print("训练时间: {}".format(end_time - start_time))
                print("训练次数: {}, Loss: {}".format(total_train_step, np.mean(train_loss)))
                print("训练次数: {}, accuracy: {}".format(total_train_step, np.mean(train_accuracy)))
                writer.add_scalar("train_loss", np.mean(train_loss), total_train_step)
                writer.add_scalar("train_accuracy", np.mean(train_accuracy), total_train_step)

        tqdm.write(
            'Epoch {0}, train_loss {1}, train_accuracy {2}'.format(epoch, np.mean(train_loss), np.mean(train_accuracy)))

    torch.save(model.state_dict(), '{0}/model_segment_picture.pth'.format(model_dir))  # 训练所有数据后，保存网络的参数
    writer.close()


