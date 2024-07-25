import torch, torchvision
from torchvision import datasets, transforms
from torch import nn, optim
from torch.nn import functional as F
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.autograd import Variable
import numpy as np
import shap


train = pd.read_csv("../input/Kannada-MNIST/train.csv", dtype = np.float32)


target = train.label.values
train = train.loc[:, train.columns != "label"].values/255

X_train, X_test, y_train, y_test = train_test_split(train, target, test_size = 0.2, random_state = 42)

X_train = torch.from_numpy(X_train)
y_train = torch.from_numpy(y_train).type(torch.LongTensor) # data type is long

X_test = torch.from_numpy(X_test)
y_test = torch.from_numpy(y_test).type(torch.LongTensor) # data type is long

batch_size = 128
num_epochs = 100

train = torch.utils.data.TensorDataset(X_train, y_train)
test = torch.utils.data.TensorDataset(X_test, y_test)

train_loader = torch.utils.data.DataLoader(train, batch_size = batch_size, shuffle = False)
test_loader = torch.utils.data.DataLoader(test, batch_size = batch_size, shuffle = False)


class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=0)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.cnn2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=0)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(32 * 4 * 4, 10)

    def forward(self, x):
        out = self.cnn1(x)
        out = self.relu1(out)
        out = self.maxpool1(out)
        out = self.cnn2(out)
        out = self.relu2(out)
        out = self.maxpool2(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        return out


# Create CNN
model = CNNModel()

# Cross Entropy Loss
error = nn.CrossEntropyLoss()

# SGD Optimizer
learning_rate = 0.1
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# CNN model training
count = 0
loss_list = []
iteration_list = []
accuracy_list = []

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):

        train = Variable(images.view(-1, 1, 28, 28))
        labels = Variable(labels)

        optimizer.zero_grad()  # Clear gradients
        outputs = model(train)  # Forward propagation
        loss = error(outputs, labels)  # Calculate softmax and cross entropy loss
        loss.backward()  # Calculating gradients
        optimizer.step()  # Update parameters

        count += 1

        if count % 50 == 0:
            # Calculate Accuracy
            correct = 0
            total = 0

            # Predict test dataset
            for images, labels in test_loader:
                test = Variable(images.view(-1, 1, 28, 28))
                outputs = model(test)  # Forward propagation
                predicted = torch.max(outputs.data, 1)[1]  # Get predictions from the maximum value
                total += len(labels)  # Total number of labels
                correct += (predicted == labels).sum()  # Total correct predictions

            accuracy = 100.0 * correct.item() / total

            # store loss and iteration
            loss_list.append(loss.data.item())
            iteration_list.append(count)
            accuracy_list.append(accuracy)
            if count % 500 == 0:
                print('Iteration: {}  Loss: {}  Accuracy: {} %'.format(count, loss.data.item(), accuracy))


batch = next(iter(test_loader))
images, _ = batch
images = images.view(-1, 1, 28, 28)

background = images[:100]
test_images= images[100:110]

e = shap.DeepExplainer(model, images)
shap_values = e.shap_values(test_images)

shap_numpy = [np.swapaxes(np.swapaxes(s, 1, -1), 1, 2) for s in shap_values]
test_numpy = np.swapaxes(np.swapaxes(test_images.numpy(), 1, -1), 1, 2)

shap.image_plot(shap_numpy, -test_numpy)
