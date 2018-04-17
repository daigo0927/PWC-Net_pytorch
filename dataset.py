from torch.utils.data import Dataset
from pathlib import Path
from itertools import islice
import numpy as np
import imageio
import torch
import random
import cv2
from flow_utils import load_flow
from abc import abstractmethod, ABCMeta


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

class BaseDataset(Dataset, metaclass = ABCMeta):
    @abstractmethod
    def __init__(self): pass
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        img1_path, img2_path, flow_path = self.samples[idx]
        img1, img2 = map(imageio.imread, (img1_path, img2_path))
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

# FlyingChairs
# ============================================================
class FlyingChairs(BaseDataset):
    def __init__(self, dataset_dir, train_or_test = 'train', color = 'rgb', shape = None):
        super(FlyingChairs, self).__init__()
        assert train_or_test in ['train', 'test']
        self.color = color
        self.shape = shape

        p = Path(dataset_dir)
        p_txt = p / (train_or_test + '.txt')
        if p_txt.exists():
            self.samples = []
            with open(p_txt, 'r') as f:
                for i in f.readlines():
                    img1, img2, flow = i.split(',')
                    flow = flow.strip()
                self.samples.append((img1, img2, flow))
        else:
            imgs = sorted(p.glob('*.ppm'))
            samples = [(str(i[0]), str(i[1]), str(i[0]).replace('img1', 'flow').replace('.ppm', '.flo')) for i in zip(imgs[::2], imgs[1::2])]
            test_ratio = 0.1
            random.shuffle(samples)
            idx = int(len(samples) * (1 - test_ratio))
            train_samples = samples[:idx]
            test_samples = samples[idx:]

            with open(p / 'train.txt', 'w') as f: f.writelines((','.join(i) + '\n' for i in train_samples))
            with open(p / 'test.txt', 'w') as f: f.writelines((','.join(i) + '\n' for i in test_samples))

            self.samples = train_samples if train_or_test == 'train' else test_samples


# FlyingThings
# ============================================================
class FlyingThings(BaseDataset):
    def __init__(self): pass

# Sintel
# ============================================================
class Sintel(BaseDataset):


    def __init__(self, text_path, mode = 'final', color = 'rgb', shape = None):
        super(Sintel, self).__init__()
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


# KITTI
# ============================================================
class KITTI(BaseDataset):

    def __init__(self):
        pass


if __name__ == '__main__':
    dataset = FlyingChairs('datasets/FlyingChairs')
    for i in range(dataset.__len__()):
        images, flow = dataset.__getitem__(i)
        print(images[0].size(), flow[0].size())