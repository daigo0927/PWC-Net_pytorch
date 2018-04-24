import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import sys

class CostVolumeLayer(nn.Module):

    def __init__(self, args):
        super(CostVolumeLayer, self).__init__()
        self.args = args

    
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
        import time
        time.sleep(60)
        f = lambda x: (x*src).sum(1)
        outputs = [f(tgt)]
        S = args.search_range
        
        for i in range(1, S + 1):
            map_up = F.pad(tgt[:,:,:-i,:], (0,0,i,0)).contiguous()
            map_down  = F.pad(tgt[:,:,i:,:], (0,0,0,i)).contiguous()
            map_left  = F.pad(tgt[:,:,:,:-i], (i,0)).contiguous()
            map_right = F.pad(tgt[:,:,:,i:], (0,i)).contiguous()
            outputs.extend(list(map(f, [map_up, map_down, map_left, map_right])))

            for j in range(1, S + 1):
                map_ul = F.pad(tgt[:,:,:-i,:-j], (j,0,i,0)).contiguous()
                map_ll = F.pad(tgt[:,:,i:,:-j], (j,0,0,i)).contiguous()
                map_ur = F.pad(tgt[:,:,:-i,j:], (0,j,i,0)).contiguous()
                map_lr = F.pad(tgt[:,:,i:,j:], (0,j,0,i)).contiguous()
                outputs.extend(list(map(f, [map_ul, map_ll, map_ur, map_lr])))


        return torch.stack(outputs, dim = 1)



class FeaturePyramidExtractor(nn.Module):

    def __init__(self, args):
        super(FeaturePyramidExtractor, self).__init__()
        self.args = args

        # it seems that pytorch can't handle modules correctly that are not in Module.__dict__?
        # so here uses setattr to add levels
        self.levels = []
        for l in range(args.num_levels):
            layer = nn.DataParallel(nn.Sequential(
                nn.Conv2d(in_channels = 3 if l == 0 else args.lv_chs[l-1], out_channels = args.lv_chs[l], kernel_size = 3, stride = 2, padding = 1),
                nn.LeakyReLU(inplace = True),
                nn.Conv2d(in_channels = args.lv_chs[l], out_channels = args.lv_chs[l], kernel_size = 3, stride = 1, padding = 1),
                nn.LeakyReLU(inplace = True)))

            setattr(self, f'level{l+1}', layer)
            self.levels.append(layer)


    def forward(self, x):
        args = self.args
        feature_pyramid = []
        out = self.levels[0](x)
        feature_pyramid.insert(0, out)
        for l in range(1, args.num_levels):
            out = self.levels[l](out)
            feature_pyramid.insert(0, out)

        return feature_pyramid
        

class OpticalFlowEstimator(nn.Module):

    def __init__(self, args, ch_in):
        super(OpticalFlowEstimator, self).__init__()
        self.args = args

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels = ch_in, out_channels = 128, kernel_size = 3, stride = 1, padding = 1, dilation = 1, groups = 1, bias = True),
            nn.LeakyReLU(inplace = True))
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = 3, stride = 1, padding = 1, dilation = 1, groups = 1, bias = True),
            nn.LeakyReLU(inplace = True))
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels = 128, out_channels = 96, kernel_size = 3, stride = 1, padding = 1, dilation = 1, groups = 1, bias = True),
            nn.LeakyReLU(inplace = True))
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels = 96, out_channels = 64, kernel_size = 3, stride = 1, padding = 1, dilation = 1, groups = 1, bias = True),
            nn.LeakyReLU(inplace = True))
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels = 64, out_channels = 32, kernel_size = 3, stride = 1, padding = 1, dilation = 1, groups = 1, bias = True),
            nn.LeakyReLU(inplace = True))
        self.conv6 = nn.Conv2d(in_channels = 32, out_channels = 2, kernel_size = 3, stride = 1, padding = 1, dilation = 1, groups = 1, bias = True)

    def forward(self, tgt, cost_volume, flow):
        args = self.args
        x = torch.cat([tgt, cost_volume, flow], dim = 1).cuda()
        
        out_conv1 = self.conv1(x)
        out_conv2 = self.conv2(out_conv1)
        out_conv3 = self.conv3(out_conv2)
        out_conv4 = self.conv4(out_conv3)
        out_feature = self.conv5(out_conv4)
        out_flow = self.conv6(out_feature)

        return out_feature, out_flow

class ContextNetwork(nn.Module):


    def __init__(self, args, ch_in):
        super(ContextNetwork, self).__init__()
        self.args = args
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels = ch_in, out_channels = 128, kernel_size = 3, stride = 1, padding = 1, dilation = 1, groups = 1, bias = True),
            nn.LeakyReLU(inplace = True))
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = 3, stride = 1, padding = 2, dilation = 2, groups = 1, bias = True),
            nn.LeakyReLU(inplace = True))
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = 3, stride = 1, padding = 4, dilation = 4, groups = 1, bias = True),
            nn.LeakyReLU(inplace = True))
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels = 128, out_channels = 96, kernel_size = 3, stride = 1, padding = 8, dilation = 8, groups = 1, bias = True),
            nn.LeakyReLU(inplace = True))
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels = 96, out_channels = 64, kernel_size = 3, stride = 1, padding = 16, dilation = 16, groups = 1, bias = True),
            nn.LeakyReLU(inplace = True))
        self.conv6 = nn.Sequential(
            nn.Conv2d(in_channels = 64, out_channels = 32, kernel_size = 3, stride = 1, padding = 1, dilation = 1, groups = 1, bias = True),
            nn.LeakyReLU(inplace = True))
        self.conv7 = nn.Conv2d(in_channels = 32, out_channels = 2, kernel_size = 3, stride = 1, padding = 1, dilation = 1, groups = 1, bias = True)

    
    def forward(self, feature, flow):
        args = self.args
        x = torch.cat([feature, flow], dim = 1)
        out_conv1 = self.conv1(x)
        out_conv2 = self.conv2(out_conv1)
        out_conv3 = self.conv3(out_conv2)
        out_conv4 = self.conv4(out_conv3)
        out_conv5 = self.conv5(out_conv4)
        out_conv6 = self.conv6(out_conv5)
        out_flow = self.conv7(out_conv6)

        return flow + out_flow