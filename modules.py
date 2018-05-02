import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import sys


def conv(args.batch_norm, in_planes, out_planes, kernel_size=3, stride=1):
    if args.batch_norm:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=False),
            nn.args.batch_norm2d(out_planes),
            nn.LeakyReLU(0.1,inplace=True)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=True),
            nn.LeakyReLU(0.1,inplace=True)
        )


class CostVolumeLayer(nn.Module):

    def __init__(self, args):
        super(CostVolumeLayer, self).__init__()
        self.args = args
        self.zeros = {}

    
    def forward(self, src, tgt):
        args = self.args

        # Version 1
        # ============================================================
        # B, C, H, W = src.size()
        # if src.size(1) >= (args.search_range*2+1)**2:
        #     output = torch.zeros_like(src)[:,:(args.search_range*2+1)**2,:,:]
        # else:
        #     output = F.pad(torch.zeros_like(src), (0,0,0,0,(args.search_range*2+1)**2 - src.size(1),0))
        # tgt = F.pad(tgt, [args.search_range]*4)
        # for i in range(args.search_range, H):
        #     for j in range(args.search_range, W):
        #         # TODO: pytorch的einsum该怎么写????
        #         tmp = [torch.matmul(src[:,:,i,j].unsqueeze(1), tgt[:,:,I,J].unsqueeze(2)) for I in range(i-args.search_range, i+args.search_range+1) for J in range(j-args.search_range, j+args.search_range+1)]
        #         tmp = torch.stack(tmp, dim = 1).squeeze()
        #         output[:,:,i,j] = tmp

        # Version 2
        # ============================================================
        # B, C, H, W = src.size()
        # if src.size(1) >= (args.search_range*2+1)**2:
        #     output = torch.zeros_like(src)[:,:(args.search_range*2+1)**2,:,:]
        # else:
        #     output = F.pad(torch.zeros_like(src), (0,0,0,0,(args.search_range*2+1)**2 - src.size(1),0))
        # tgt = F.pad(tgt, [args.search_range]*4)
        # for i in range(args.search_range, H):
        #     for j in range(args.search_range, W):\
        #         output[:,:,i,j] = torch.matmul(src[:,:,i,j].unsqueeze(1), tgt[:,:,i-args.search_range:i+args.search_range+1,j-args.search_range:j+args.search_range+1].contiguous().view(B, C, -1)).squeeze(1)

        # Version 3
        # ============================================================
        # tgt_neigh = [tgt]
        # for i in range(1, args.search_range + 1):
        #     map_up    = torch.zeros_like(tgt); map_up[:,:,i:,:]     = tgt[:,:,:-i,:]
        #     map_down  = torch.zeros_like(tgt); map_down[:,:,:-i,:]  = tgt[:,:,i:,:]
        #     map_left  = torch.zeros_like(tgt); map_left[:,:,:,i:]   = tgt[:,:,:,:-i]
        #     map_right = torch.zeros_like(tgt); map_right[:,:,:,:-i] = tgt[:,:,:,i:]
        #     tgt_neigh.extend([map_up, map_down, map_left, map_right])

        #     for j in range(1, args.search_range + 1):
        #         map_ul = torch.zeros_like(tgt); map_ul[:,:,i:,j:]   = tgt[:,:,:-i,:-j]
        #         map_ll = torch.zeros_like(tgt); map_ll[:,:,:-i,j:]  = tgt[:,:,i:,:-j]
        #         map_ur = torch.zeros_like(tgt); map_ur[:,:,i:,:-j]  = tgt[:,:,:-i,j:]
        #         map_lr = torch.zeros_like(tgt); map_lr[:,:,:-i,:-j] = tgt[:,:,i:,j:]
        #         tgt_neigh.extend([map_ul, map_ll, map_ur, map_lr])

        # tgt_neigh = torch.stack(tgt_neigh, dim = 2)
        
        # output = (src.unsqueeze(dim = 2) * tgt_neigh).sum(dim = 1)


        # Version 4
        # ============================================================
        S = args.search_range
        B, C, H, W = src.size()
        if H not in self.zeros:
            self.zeros[H] = torch.zeros((B, (S*2+1)**2, H, W)).to(args.device)
        output = torch.zeros_like(self.zeros[H])
        output[:,0] = (tgt*src).sum(1)

        I = 1
        for i in range(1, S + 1):
            # tgt下移i像素并补0, src与之对应的部分为i之后的像素, output的上i个像素为0
            output[:,I,i:,:] = (tgt[:,:,:-i,:] * src[:,:,i:,:]).sum(1); I += 1
            output[:,I,:-i,:] = (tgt[:,:,i:,:] * src[:,:,:-i,:]).sum(1); I += 1
            output[:,I,:,i:] = (tgt[:,:,:,:-i] * src[:,:,:,i:]).sum(1); I += 1
            output[:,I,:,:-i] = (tgt[:,:,:,i:] * src[:,:,:,:-i]).sum(1); I += 1

            for j in range(1, S + 1):
                output[:,I,i:,j:] = (tgt[:,:,:-i,:-j] * src[:,:,i:,j:]).sum(1); I += 1
                output[:,I,:-i,:-j] = (tgt[:,:,i:,j:] * src[:,:,:-i,:-j]).sum(1); I += 1
                output[:,I,i:,:-j] = (tgt[:,:,:-i,j:] * src[:,:,i:,:-j]).sum(1); I += 1
                output[:,I,:-i,j:] = (tgt[:,:,i:,:-j] * src[:,:,:-i,j:]).sum(1); I += 1

        return output / C



class FeaturePyramidExtractor(nn.Module):

    def __init__(self, args):
        super(FeaturePyramidExtractor, self).__init__()
        self.args = args

        self.convs = []
        for l in range(args.num_levels):
            layer = nn.Sequential(
                conv(args.batch_norm, 3 if l == 0 else args.lv_chs[layer_idx - 1], args.lv_chs[layer_idx], stride = 2),
                conv(args.batch_norm, args.lv_chs[layer_idx], args.lv_chs[layer_idx])
            )
            self.add_module(f'Feature(Lv{layer_idx + 1})', layer)
            self.convs.append(layer)


    def forward(self, x):
        feature_pyramid = []
        for conv in self.convs:
            x = conv(x); feature_pyramid.append(x)

        return feature_pyramid
        

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
        return self.convs(x)

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