import os
import torch
import glob
import cv2

from torch.utils.data import Dataset


class MNIST(Dataset):
    def __init__(self, df):
        """
        Pytorch MNIST dataset class. Designed to use with sample MNIST dataset
        provided in Colab.

        Args:
            df (pd.DataFrame): MNIST dataset in csv file format from colab.
        """
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        data = self.df.iloc[index].tolist()
        image = torch.tensor(data[1:])
        image = image.view(1, 28, 28) / 255
        # normalize to [-1, 1]
        image = image * 2 - 1

        return image


class CustomDataset(Dataset):
    def __init__(self, folder_path, img_size=64):
        """
        Pytorch Custom Dataset class
        Args:
            folder_path (str): path to folder where images are stored.
            img_size (int): width and height of returned images
        """
        self.images = glob.glob(os.path.join(folder_path, '*'))
        self.img_size = img_size

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        path = self.images[index]
        image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, dsize=(64, 64))
        image = image / 255
        # normalize to [-1, 1]
        image = image * 2 - 1
        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)

        return image