import torch.nn as nn
from model import Resnet_50
import matplotlib.pyplot as plt
import torch
from PIL import Image
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.transforms.functional import to_pil_image
from matplotlib import colormaps
import numpy as np
import PIL
from types import MethodType


model_path = "D:\\project\\city_AI\\origin_model.pth"

# Define ResNet50 model architecture
resnet_model = Resnet_50(num_classes=4)

# Load saved weights from .pkl file
resnet_model = nn.DataParallel(resnet_model)
resnet_model.load_state_dict(torch.load(model_path, map_location=torch.device('cuda')))

# Put it in evaluation mode for inference
resnet_model.eval()

# Defines two global scope variables to store our gradients and activations
gradients = None
activations = None


def backward_hook(module, grad_input, grad_output):
    global gradients
    print('Backward hook running...')
    gradients = grad_output
    print(f'Gradients size: {gradients[0].size()}')


def forward_hook(module, input, output):
    global activations
    print('Forward hook running...')
    activations = output
    print(f'Activations size: {activations.size()}')


backward_handle = resnet_model.module.resnet50.layer4[-1].register_full_backward_hook(hook=backward_hook)
forward_handle = resnet_model.module.resnet50.layer4[-1].register_forward_hook(hook=forward_hook)


# Output the model's last layer for examination, identifying the last convolutional layer and its activation function
print(resnet_model.module.resnet50.layer4[-1])

img_path = "D:\\project\\city_AI\\test.jpg"
image = Image.open(img_path).convert('RGB')
image_numpy = np.array(image)

# Define data transforms for the input image
transform_test = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize to 256x256
    # transforms.RandomCrop((500, 224)),  # Random crop to 224x224
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

img_tensor = transform_test(image)  # Convert image to tensor
img_tensor = img_tensor.to(resnet_model.src_device_obj)
img_tensor_input = img_tensor.unsqueeze(0)
output = resnet_model(img_tensor_input)
resnet_model(img_tensor.unsqueeze(0)).backward(torch.tensor([[0, 1, 0, 0]]).to(resnet_model.src_device_obj))

pooled_gradients = torch.mean(gradients[0], dim=[0, 2, 3])

# Weight the channels by corresponding gradients
for i in range(activations.size()[1]):
    activations[:, i, :, :] *= pooled_gradients[i]

# Average the channels of the activations
heatmap = torch.mean(activations, dim=1).squeeze()

# ReLU on top of the heatmap
heatmap = F.relu(heatmap)

# Normalize the heatmap
heatmap /= torch.max(heatmap)

# Draw the heatmap
plt.matshow(heatmap.cpu().detach())

# Create a figure and plot the first image
fig, ax = plt.subplots()
ax.axis('off')  # Remove the axis markers

# Plot the original image
ax.imshow(to_pil_image(img_tensor, mode='RGB'))

# Resize the heatmap to the same size as the input image and define a resample algorithm for increasing image resolution
# We need heatmap.detach() because it can't be converted to a numpy array while requiring gradients
overlay = to_pil_image(heatmap.detach(), mode='F').resize((256, 256), resample=PIL.Image.BICUBIC)

# Apply any colormap you want
cmap = colormaps['jet']
overlay = (255 * cmap(np.asarray(overlay) ** 2)[:, :, :3]).astype(np.uint8)

# Plot the heatmap on the same axes, but with alpha < 1 (this defines the transparency of the heatmap)
ax.imshow(overlay, alpha=0.4, interpolation='nearest')

# Show the plot
plt.show()

# Remove the hooks
backward_handle.remove()
forward_handle.remove()
