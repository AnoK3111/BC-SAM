import os
import random
import h5py
import gzip
import numpy as np
import pickle
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
from einops import repeat
from icecream import ic
from torchvision import transforms


def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


class RandomGenerator(object):
    def __init__(self, output_size, low_res):
        self.output_size = output_size
        self.low_res = low_res

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        x, y = image.shape[0:2]
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y,1), order=3)  # why not 3?
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        label_h, label_w = label.shape
        low_res_label = zoom(label, (self.low_res[0] / label_h, self.low_res[1] / label_w), order=0)
        cropped_img=image.copy()
        cropped_img[~np.stack([label.astype(bool)]*3, axis=2).reshape(self.output_size[0],self.output_size[1],3)]=128
        cropped_img = torch.from_numpy(cropped_img.astype(np.float32)/255)
        cropped_img=cropped_img.permute(2,0,1)
        image = torch.from_numpy(image.astype(np.float32))
        image=image.permute(2,0,1)
        label = torch.from_numpy(label.astype(np.float32))
        low_res_label = torch.from_numpy(low_res_label.astype(np.float32))
        sample = {'image': image, 'label': label.long(), 'low_res_label': low_res_label.long(), 'cropped_img':cropped_img}
        return sample


class Bloodcell_dataset(Dataset):
    def __init__(self, base_dir, transform=None):
        self.transform = transform  # using transform in torch!
        self.data_dir = base_dir
        samples={}
        with gzip.open(base_dir, "rb") as f:
            data = pickle.load(f)
            samples.update(data)
        self.samples=samples
        self.keys=list(samples.keys())
        means=np.zeros((3,))
        stds=np.zeros((3,))
        for sample in samples:
            means+=np.mean(samples[sample]['image'], axis=(0, 1))
            stds+=np.std(samples[sample]['image'], axis=(0, 1))
        self.means=means/len(self.keys)
        self.stds=stds/len(self.keys)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        key=self.keys[idx]
        sample = {'image': self.samples[key]['image'].copy(), 'label': self.samples[key]['label'].copy()}
        if self.transform:
            sample = self.transform(sample)
        sample['case_name'] = str(key)
        return sample
    
    def get_mean_std(self):
        return self.means,self.stds





if __name__=='__main__':
    data_cell=Bloodcell_dataset('/home/anoke/python/SAMed/datasets/Bloodcell/Bloodcell.dat.gz',transform=transforms.Compose(
                                   [RandomGenerator(output_size=[224, 224], low_res=[56, 56])]))
    import matplotlib.pyplot as plt
    print(data_cell.samples[data_cell.keys[123]]['image'].shape)
    print(data_cell[123]['low_res_label'].shape)
    print(data_cell.get_mean_std())
    #plt.imshow(np.rollaxis(data_cell[123]['image'].cpu().numpy().astype(np.uint8),0,3))
    #plt.show()
    
    



    
