import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class WarpingLayer(nn.Module):

    def __init__(self, args):
        super(WarpingLayer, self).__init__()
        self.args = args


    def forward(self, x, flow):
        args = self.args
        # build coord matrix
        torchHorizontal = torch.linspace(-1.0, 1.0, x.size(3)).view(1, 1, 1, x.size(3)).expand(x.size(0), 1, x.size(2), x.size(3))
        torchVertical = torch.linspace(-1.0, 1.0, x.size(2)).view(1, 1, x.size(2), 1).expand(x.size(0), 1, x.size(2), x.size(3))


        grid = torch.cat([torchHorizontal, torchVertical], 1)
        if not args.no_cuda: grid = grid.cuda()

        grid = Variable(data = grid, volatile = not self.training)

        # print(x.size(), grid.size(), flow.size())

        # variableFlow = torch.cat([ variableFlow[:, 0:1, :, :] / ((variableInput.size(3) - 1.0) / 2.0), variableFlow[:, 1:2, :, :] / ((variableInput.size(2) - 1.0) / 2.0) ], 1)
        return F.grid_sample(x, (grid + flow).permute(0, 2, 3, 1))


class CostVolumeLayer(nn.Module):

    def __init__(self, args):
        super(CostVolumeLayer, self).__init__()
        self.args = args

    
    def forward(self, src, tgt):
        args = self.args
        tgt = F.pad(tgt, [args.search_range]*4)
        B, C, H, W = src.size()
        if src.size(1) >= (args.search_range*2+1)**2:
            output = torch.zeros_like(src)[:,:(args.search_range*2+1)**2,:,:]
        else:
            output = F.pad(torch.zeros_like(src), (0,0,0,0,(args.search_range*2+1)**2 - src.size(1),0))

        import time

        # t = time.time()
        # # TODO: so slow! find the batch dot way
        # for i in range(args.search_range, H):
        #     for j in range(args.search_range, W):
        #         # TODO: pytorch的einsum该怎么写????
        #         tmp = [torch.matmul(src[:,:,i,j].unsqueeze(1), tgt[:,:,I,J].unsqueeze(2)) for I in range(i-args.search_range, i+args.search_range+1) for J in range(j-args.search_range, j+args.search_range+1)]
        #         tmp = torch.stack(tmp, dim = 1).squeeze()
        #         output[:,:,i,j] = tmp
        # print('v1:', time.time()-t)
        # t = time.time()

        t = time.time()
        for i in range(args.search_range, H):
            for j in range(args.search_range, W):\
                output[:,:,i,j] = torch.matmul(src[:,:,i,j].unsqueeze(1), tgt[:,:,i-args.search_range:i+args.search_range+1,j-args.search_range:j+args.search_range+1].contiguous().view(B, C, -1)).squeeze(1)
        print('v2:', time.time()-t)

        t = time.time()

        tgt_neigh = [tgt]
        for i in range(1, args.search_range + 1):
            map_left  = torch.zeros_like(tgt); map_left[:,:,i:, :]   = tgt[:,:,:-i, :]
            map_right = torch.zeros_like(tgt); map_right[:,:,:-i, :] = tgt[:,:,i:, :]
            map_up    = torch.zeros_like(tgt); map_up[:,:,:, i:]     = tgt[:,:,:, :-i]
            map_down  = torch.zeros_like(tgt); map_down[:,:,:, :-i]  = tgt[:,:,:, i:]
            tgt_neigh.extend([map_left, map_right, map_up, map_down])
        
        tgt_neigh = torch.stack(tgt_neigh, dim = 2)
        

        output = (src.unsqueeze(2) * tgt_neigh).sum(dim = 1)
        print('v3:', time.time()-t)

        return output



class FeaturePyramidExtractor(nn.Module):

    def __init__(self, args):
        super(FeaturePyramidExtractor, self).__init__()
        self.args = args

        self.level1 = nn.Sequential(
            nn.Conv2d(in_channels = 3, out_channels = 16, kernel_size = 3, stride = 2, padding = 1, dilation = 1, groups = 1, bias = True),
            nn.LeakyReLU(inplace = True),
            nn.Conv2d(in_channels = 16, out_channels = 16, kernel_size = 3, stride = 1, padding = 1, dilation = 1, groups = 1, bias = True),
            nn.LeakyReLU(inplace = True))
        self.level2 = nn.Sequential(
            nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = 3, stride = 2, padding = 1, dilation = 1, groups = 1, bias = True),
            nn.LeakyReLU(inplace = True),
            nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = 3, stride = 1, padding = 1, dilation = 1, groups = 1, bias = True),
            nn.LeakyReLU(inplace = True))
        self.level3 = nn.Sequential(
            nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 3, stride = 2, padding = 1, dilation = 1, groups = 1, bias = True),
            nn.LeakyReLU(inplace = True),
            nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, stride = 1, padding = 1, dilation = 1, groups = 1, bias = True),
            nn.LeakyReLU(inplace = True))
        self.level4 = nn.Sequential(
            nn.Conv2d(in_channels = 64, out_channels = 96, kernel_size = 3, stride = 2, padding = 1, dilation = 1, groups = 1, bias = True),
            nn.LeakyReLU(inplace = True),
            nn.Conv2d(in_channels = 96, out_channels = 96, kernel_size = 3, stride = 1, padding = 1, dilation = 1, groups = 1, bias = True),
            nn.LeakyReLU(inplace = True))
        self.level5 = nn.Sequential(
            nn.Conv2d(in_channels = 96, out_channels = 128, kernel_size = 3, stride = 2, padding = 1, dilation = 1, groups = 1, bias = True),
            nn.LeakyReLU(inplace = True),
            nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = 3, stride = 1, padding = 1, dilation = 1, groups = 1, bias = True),
            nn.LeakyReLU(inplace = True))
        self.level6 = nn.Sequential(
            nn.Conv2d(in_channels = 128, out_channels = 192, kernel_size = 3, stride = 2, padding = 1, dilation = 1, groups = 1, bias = True),
            nn.LeakyReLU(inplace = True),
            nn.Conv2d(in_channels = 192, out_channels = 192, kernel_size = 3, stride = 1, padding = 1, dilation = 1, groups = 1, bias = True),
            nn.LeakyReLU(inplace = True))

        self.levels = [self.level1, self.level2, self.level3, self.level4, self.level5, self.level6]


    def forward(self, x):
        args = self.args
        feature_pyramid = []
        out = self.levels[0](x)
        feature_pyramid.insert(0, out)
        for l in range(1, args.num_levels):
            out = self.levels[l](out)
            feature_pyramid.insert(0, out)

        # for i in feature_pyramid:
        #     print(i.size())
        # quit()
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
        # print(tgt.size(), cost_volume.size(), flow.size())
        x = torch.cat([tgt, cost_volume, flow], dim = 1).cuda()
        
        out_conv1 = self.conv1(x)
        out_conv2 = self.conv2(out_conv1)
        out_conv3 = self.conv3(out_conv2)
        out_conv4 = self.conv4(out_conv3)
        out_feature = self.conv5(out_conv4)
        out_flow = self.conv6(out_feature)

        # print(x.size(),
        # out_conv1.size(), out_conv2.size(), out_conv3.size(), out_conv4.size(), 
        # out_feature.size(), out_flow.size(),
        # sep = '\n')
        
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

        # print(x.size(),
        # out_conv1.size(), out_conv2.size(), out_conv3.size(), out_conv4.size(),
        # out_conv5.size(), out_conv6.size(),
        # out_flow.size(),
        # sep = '\n')


        return flow + out_flow