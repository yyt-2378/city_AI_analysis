from PIL import Image
import torch.utils.data as data


class CustomDataset(data.Dataset):
    def __init__(self, data_path, transform=None):
        self.data_path = data_path
        self.transform = transform
        self.train_labels = []

        # load data list
        with open(data_path, 'r') as f:
            self.data_list = f.readlines()

        with open(data_path, 'r') as f:
            for line in f:
                label = int(line.strip()[-1])
                self.train_labels.append(label)

    def __getitem__(self, index):
        # load image and label from data list
        data = self.data_list[index].strip().split(',')
        image_path = data[0]
        label = self.train_labels[index]

        # load image
        with open(image_path, 'rb') as f:
            image = Image.open(f).convert('RGB')

        # apply transform if specified
        if self.transform is not None:
            image = self.transform(image)

        return image, label, index

    def __len__(self):
        return len(self.data_list)
