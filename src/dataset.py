from torchvision import datasets
from torch.utils import data
import pickle
from os.path import exists, join
import numpy as np
from PIL import Image


class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """

    # override the __getitem__ method. this is the method dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path


class FeatDataset(data.Dataset):
    def __init__(self, key_path, feat_dir, gt_path):
        assert exists(key_path) and exists(feat_dir) and exists(gt_path)
        self.keys = pickle.load(open(key_path, 'rb'))
        self.gts = pickle.load(open(gt_path, 'rb'))
        self.feat_dir = feat_dir

    def __getitem__(self, index):
        key = self.keys[index]
        target = self.gts[key]
        feat_path = join(self.feat_dir, key + '.npz')
        feat = np.load(feat_path)['arr_0']
        return feat, target, key

    def __len__(self):
        return len(self.keys)


class ImageData(data.Dataset):
    def __init__(self, key_path, image_dir, gt_path, transform=None):
        assert exists(key_path) and exists(image_dir) and exists(gt_path)
        self.keys = pickle.load(open(key_path, 'rb'))
        self.gts = pickle.load(open(gt_path, 'rb'))
        self.image_dir = image_dir
        self.transform = transform

    def __getitem__(self, index):
        key = self.keys[index]
        target = self.gts[key]
        image_path = join(self.image_dir, key + '.jpg')
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, target, key

    def __len__(self):
        return len(self.keys)
