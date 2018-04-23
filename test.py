import torch
import torch.nn.functional as F
from torch.autograd import Variable

B, C, H, W = 8, 128, 48, 48
S = 4


def v5_cpu(src, tgt):
    f = lambda x: (x*src).sum(1)
    outputs = [f(tgt)]
    
    for i in range(1, S + 1):
        map_up = F.pad(tgt[:,:,:-i,:], (0,0,0,i))
        map_down  = F.pad(tgt[:,:,i:,:], (0,0,0,i))
        map_left  = F.pad(tgt[:,:,:,:-i], (i,0))
        map_right = F.pad(tgt[:,:,:,i:], (i,0))
        outputs.extend(list(map(f, [map_up, map_down, map_left, map_right])))

        for j in range(1, S + 1):
            map_ul = F.pad(tgt[:,:,:-i,:-j], (j,0,i,0))
            map_ll = F.pad(tgt[:,:,i:,:-j], (j,0,0,i))
            map_ur = F.pad(tgt[:,:,:-i,j:], (0,j,0,i))
            map_lr = F.pad(tgt[:,:,i:,j:], (0,j,i,0))
            outputs.extend(list(map(f, [map_ul, map_ll, map_ur, map_lr])))


def v5_gpu(src, tgt):
    f = lambda x: (x*src).sum(1)
    outputs = [f(tgt)]
    
    for i in range(1, S + 1):
        map_up = F.pad(tgt[:,:,:-i,:], (0,0,0,i))
        map_down  = F.pad(tgt[:,:,i:,:], (0,0,0,i))
        map_left  = F.pad(tgt[:,:,:,:-i], (i,0))
        map_right = F.pad(tgt[:,:,:,i:], (i,0))
        outputs.extend(list(map(f, [map_up, map_down, map_left, map_right])))

        for j in range(1, S + 1):
            map_ul = F.pad(tgt[:,:,:-i,:-j], (j,0,i,0))
            map_ll = F.pad(tgt[:,:,i:,:-j], (j,0,0,i))
            map_ur = F.pad(tgt[:,:,:-i,j:], (0,j,0,i))
            map_lr = F.pad(tgt[:,:,i:,j:], (0,j,i,0))
            outputs.extend(list(map(f, [map_ul, map_ll, map_ur, map_lr])))