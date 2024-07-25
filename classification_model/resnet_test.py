from Dataset_city import Dataset
import torchvision.models as models
from torch.autograd import Variable
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # 防止打印图片出错


data_dir = '/Classification/Dataset'
model_file = './model/model_segment_picture.pth'

# Define ResNet50 model architecture
resnet = models.resnet50()

# Load saved weights from .pkl file
model = nn.DataParallel(resnet)
model.load_state_dict(torch.load(model_file))
model.eval()

datafile = Dataset('test', data_dir)
# Test the model
correct = 0
total = 0

for index in range(datafile.data_size):
    img, labels = datafile.__getitem__(index)
    img = img.unsqueeze(0)
    img = Variable(img)

    outputs = model(img)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += int(predicted == labels)

accuracy = 100 * correct / total
print('Accuracy on test set: %d %%' % accuracy)