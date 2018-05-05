import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import sys

from utils import get_grid


def conv(batch_norm, in_planes, out_planes, kernel_size=3, stride=1):
    if batch_norm:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.LeakyReLU(0.1,inplace=True)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=True),
            nn.LeakyReLU(0.1,inplace=True)
        )


class WarpingLayer(nn.Module):
    
    def __init__(self, args):
        super(WarpingLayer, self).__init__()
        self.args = args
    
    def forward(self, x, flow):
        args = self.args
        grid = (get_grid(x).to(args.device) + flow).permute(0, 2, 3, 1)
        x_warp = F.grid_sample(x, grid)
        return x_warp


class CostVolumeLayer(nn.Module):

    def __init__(self, args):
        super(CostVolumeLayer, self).__init__()
        self.args = args
        self.search_range = args.search_range

    
    def forward(self, src, tgt):
        args = self.args

        shape = list(src.size()); shape[1] = (self.search_range * 2 + 1) ** 2
        output = torch.zeros(shape).to(args.device)
        output[:,0] = (tgt*src).sum(1)

        I = 1
        for i in range(1, self.search_range + 1):
            # tgt下移i像素并补0, src与之对应的部分为i之后的像素, output的上i个像素为0
            output[:,I,i:,:] = (tgt[:,:,:-i,:] * src[:,:,i:,:]).sum(1); I += 1
            output[:,I,:-i,:] = (tgt[:,:,i:,:] * src[:,:,:-i,:]).sum(1); I += 1
            output[:,I,:,i:] = (tgt[:,:,:,:-i] * src[:,:,:,i:]).sum(1); I += 1
            output[:,I,:,:-i] = (tgt[:,:,:,i:] * src[:,:,:,:-i]).sum(1); I += 1

            for j in range(1, self.search_range + 1):
                output[:,I,i:,j:] = (tgt[:,:,:-i,:-j] * src[:,:,i:,j:]).sum(1); I += 1
                output[:,I,:-i,:-j] = (tgt[:,:,i:,j:] * src[:,:,:-i,:-j]).sum(1); I += 1
                output[:,I,i:,:-j] = (tgt[:,:,:-i,j:] * src[:,:,i:,:-j]).sum(1); I += 1
                output[:,I,:-i,j:] = (tgt[:,:,i:,:-j] * src[:,:,:-i,j:]).sum(1); I += 1

        return output / shape[1]


class FeaturePyramidExtractor(nn.Module):

    def __init__(self, args):
        super(FeaturePyramidExtractor, self).__init__()
        self.args = args

        self.convs = []
        for l in range(args.num_levels - 1):
            layer = nn.Sequential(
                conv(args.batch_norm, 3 if l == 0 else args.lv_chs[l - 1], args.lv_chs[l], stride = 2),
                conv(args.batch_norm, args.lv_chs[l], args.lv_chs[l])
            )
            self.add_module(f'Feature(Lv{l + 1})', layer)
            self.convs.append(layer)


    def forward(self, x):
        feature_pyramid = []
        for conv in self.convs:
            x = conv(x); feature_pyramid.append(x)

        return feature_pyramid[::-1]
        

class OpticalFlowEstimator(nn.Module):

    def __init__(self, args, ch_in):
        super(OpticalFlowEstimator, self).__init__()
        self.args = args

        self.convs = nn.Sequential(
            conv(args.batch_norm, ch_in, 128),
            conv(args.batch_norm, 128, 128),
            conv(args.batch_norm, 128, 96),
            conv(args.batch_norm, 96, 64),
            conv(args.batch_norm, 64, 32),
            nn.Conv2d(in_channels = 32, out_channels = 2, kernel_size = 3, stride = 1, padding = 1, dilation = 1, groups = 1, bias = True)
        )

    def forward(self, x):
        args = self.args
        if args.flow_norm:
            output = F.tanh(self.convs(x))
            new_output = torch.zeros_like(output)
            new_output[:,0,:,:] = output[:,0,:,:] * x.size(3)
            new_output[:,1,:,:] = output[:,1,:,:] * x.size(2)
        else:
            output = self.convs(x)
        return output


class ContextNetwork(nn.Module):


    def __init__(self, args, ch_in):
        super(ContextNetwork, self).__init__()
        self.args = args

        self.convs = nn.Sequential(
            nn.Conv2d(in_channels = ch_in, out_channels = 128, kernel_size = 3, stride = 1, padding = 1, dilation = 1, groups = 1, bias = True),
            nn.LeakyReLU(inplace = True),
            nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = 3, stride = 1, padding = 2, dilation = 2, groups = 1, bias = True),
            nn.LeakyReLU(inplace = True),
            nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = 3, stride = 1, padding = 4, dilation = 4, groups = 1, bias = True),
            nn.LeakyReLU(inplace = True),
            nn.Conv2d(in_channels = 128, out_channels = 96, kernel_size = 3, stride = 1, padding = 8, dilation = 8, groups = 1, bias = True),
            nn.LeakyReLU(inplace = True),
            nn.Conv2d(in_channels = 96, out_channels = 64, kernel_size = 3, stride = 1, padding = 16, dilation = 16, groups = 1, bias = True),
            nn.LeakyReLU(inplace = True),
            nn.Conv2d(in_channels = 64, out_channels = 32, kernel_size = 3, stride = 1, padding = 1, dilation = 1, groups = 1, bias = True),
            nn.LeakyReLU(inplace = True),
            nn.Conv2d(in_channels = 32, out_channels = 2, kernel_size = 3, stride = 1, padding = 1, dilation = 1, groups = 1, bias = True)
        )
    
    def forward(self, x):
        return self.convs(x)