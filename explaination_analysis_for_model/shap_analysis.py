import torch.nn as nn
from model import Resnet_50
import shap
import torchvision.transforms as transforms
import torch.utils.data
import torch.utils.data.distributed
import numpy as np
from Dataset_city import Dataset
import matplotlib.pyplot as plt
from skimage.segmentation import slic
from PIL import Image


# define a function that depends on a binary mask representing if an image region is hidden
def mask_image(zs, segmentation, image, background=None):
    # zs is the existing segmentation array. if == 1 means the seg appears if ==0 means it disappear
    if background is None:
        background = np.array([-1.0615, -1.0615, -1.0615])
    out = np.zeros((zs.shape[0], image.shape[0], image.shape[1], image.shape[2]))
    for i in range(zs.shape[0]):
        out[i, :, :, :] = image
        for j in range(zs.shape[1]):
            if zs[i, j] == 0:
                out[i][segmentation == j, :] = background
    out = torch.from_numpy(out).cpu()
    out = out.transpose(1, 3)
    out = out.type(torch.FloatTensor)
    return out

criterion = nn.CrossEntropyLoss()


class Evaluate_model(object):
    def __init__(self,model,ground_label):
        self.model = model.to('cuda')
        self.ground_label = ground_label

    def get_loss(self, imgs):
        self.model.eval()
        losses = []

        for img in imgs:
            out = self.model(img.unsqueeze(0))  # Add an extra dimension for batch

            ground_label = torch.tensor(self.ground_label).to('cuda')

            loss = criterion(out, ground_label)
            losses.append(loss.detach().cpu().numpy())

        losses = np.array(losses)

        return losses


def f(z):
    out = model.get_loss(mask_image(z, segments_slic, test_images_numpy, None))
    return out


# plot our explanations
def fill_segmentation(values, segmentation):
    out = np.zeros(segmentation.shape)
    for i in range(len(values[0])):
        out[segmentation == i] = values[0,i]
    return out


if __name__ == '__main__':
    model_path = "/home/lijiali/projects/city_AI/Model files/model_segment_picture.pth"
    # test_dir =

    # Define ResNet50 model architecture
    resnet_model = Resnet_50(num_classes=4)

    # Load saved weights from .pkl file
    resnet_model = nn.DataParallel(resnet_model)
    resnet_model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))


    # Data loading code
    # class_names = ["BOATQUAY", "CHINATOWN", "ImageKampong", "ImageLittleindia"]
    # Split the lines by whitespace to get the file name and label
    test_data_dir = []
    with open(test_dir, 'r') as f:
        lines = f.readlines()  # 读取所有行

    for line in lines:
        filename, label = line.strip().split(',')  # 去除末尾的换行符，并按空格分割
        test_data_dir.append((filename, int(label)))  # 将文件名和标签存储为一个元组，添加到data列表中

    for file in test_data_dir:  # 遍历
        test_file_name = str(file[0])
        label_x = file[1]
        ground_label = [label_x]
        model = Evaluate_model(model=resnet_model, ground_label=ground_label)
        image = Image.open(test_file_name).convert('RGB')
        # Define data transforms for the input image
        # Define data transforms for training data
        transform_test = transforms.Compose([
            transforms.Resize((256, 256)),  # 统一缩放到256x256
            # transforms.RandomCrop((224, 224)),  # 随机裁剪到224x224
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        img_tensor = transform_test(image)  # Convert image to tensor
        img_tensor = img_tensor / img_tensor.max()
        test_images = img_tensor
        test_images_switch = test_images.permute(1, 2, 0)
        test_images_numpy = test_images_switch.cpu().detach().numpy()

        # segment the image so we don't have to explain every pixel
        segments_slic = slic(test_images_numpy, n_segments=2000, compactness=0.1, sigma=2)
        h, w = segments_slic.shape
        new_map = np.zeros_like(segments_slic)

        explainer = shap.KernelExplainer(f, np.zeros((1, 2000)))
        shap_values = explainer.shap_values(np.ones((1, 2000)), nsamples=10000)

        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))
        axes[0].imshow(test_images_numpy)
        axes[0].axis('off')
        max_val = np.max([np.max(np.abs(shap_values[i])) for i in range(len(shap_values))])

        axes[1].imshow(test_images_numpy)
        m = fill_segmentation(shap_values, segments_slic)
        im = axes[1].imshow(m, cmap="bwr", vmin=0, vmax=max_val, alpha=0.5)
        axes[1].axis('off')

        cb = fig.colorbar(im, ax=axes.ravel().tolist(), label="SHAP value", orientation="horizontal", aspect=60)
        cb.outline.set_visible(False)

        plt.savefig(r"./test_shap_high_precise_2000.png", dpi=1200)