from torch.utils.data import Dataset
from pathlib import Path
from itertools import islice
import numpy as np
import imageio
import torch
import random
import cv2
from flow_utils import load_flow


class StaticRandomCrop(object):
    def __init__(self, image_size, crop_size):
        self.th, self.tw = crop_size
        h, w = image_size
        self.h1 = random.randint(0, h - self.th)
        self.w1 = random.randint(0, w - self.tw)

    def __call__(self, img):
        return img[self.h1:(self.h1+self.th), self.w1:(self.w1+self.tw),:]


class StaticCenterCrop(object):
    def __init__(self, image_size, crop_size):
        self.th, self.tw = crop_size
        self.h, self.w = image_size
    def __call__(self, img):
        return img[(self.h-self.th)/2:(self.h+self.th)/2, (self.w-self.tw)/2:(self.w+self.tw)/2,:]


def window(seq, n=2):
    "Returns a sliding window (of width n) over data from the iterable"
    "   s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...                   "
    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) == n:
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result


class MPISintel(Dataset):


    def __init__(self, text_path, mode = 'final', color = 'rgb', shape = None):
        super(MPISintel, self).__init__()
        self.mode = mode
        self.color = color
        self.shape = shape

        root = Path('mpi/training/' + mode)
        self.samples = []
        with open(text_path, 'r') as f:
            paths = f.readlines()
        for path in paths:
            path = path.strip()
            img_paths = sorted((root / path).iterdir())

            for i in window(img_paths, 2):
                self.samples.append(i)
        # root = Path(mpi_path) / 'training'
        # img_dir = root / mode
        # flow_dir = root / 'flow'

        # l = list(img_dir.iterdir())
        # l1 = l[:20]
        # l2 = l[20:]
        # with open('data_train.txt', 'w') as f:
        #     f.writelines((str(i) + '\n' for i in l1))
        
        # with open('data_test.txt', 'w') as f:
        #     f.writelines((str(i) + '\n' for i in l2))

    def __getitem__(self, idx):
        img_path1, img_path2 = self.samples[idx]
        img1, img2 = imageio.imread(str(img_path1)), imageio.imread(str(img_path2))

        flow_path = str(img_path1).replace('.png', '.flo').replace(self.mode, 'flow')
        flow = load_flow(flow_path)
        if self.color == 'gray':
            img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)[:,:,np.newaxis]
            img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)[:,:,np.newaxis]

        images = [img1, img2]
        if self.shape is not None:
            cropper = StaticRandomCrop(img1.shape[:2], self.shape)
            images = list(map(cropper, images))
            flow = cropper(flow)
        images = np.array(images).transpose(3,0,1,2)
        flow = flow.transpose(2,0,1)

        images = torch.from_numpy(images.astype(np.float32))
        flow = torch.from_numpy(flow.astype(np.float32))

        return [images], [flow]
    

    def __len__(self):
        return len(self.samples)


if __name__ == '__main__':
    dataset = MPISintel('data_train.txt', shape = (384,768))
    for i in range(dataset.__len__()):
        images, flow = dataset.__getitem__(i)
        print(images[0].size(), flow[0].size())