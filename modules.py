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

		
        # variableFlow = torch.cat([ variableFlow[:, 0:1, :, :] / ((variableInput.size(3) - 1.0) / 2.0), variableFlow[:, 1:2, :, :] / ((variableInput.size(2) - 1.0) / 2.0) ], 1)
        return F.grid_sample(x, grid + flow)


class CostVolumeLayer(nn.Module):

    def __init__(self, args):
        super(CostVolumeLayer, self).__init__()
        self.args = args

    
    def forward(self, src, tgt):
        args = self.args
        print(src.size(), tgt.size())
        pass



class FeaturePyramidExtractor(nn.Module):

    def __init__(self, args):
        super(FeaturePyramidExtractor, self).__init__()
        self.args = args


        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels = 3, out_channels = 16, kernel_size = 3, stride = 2, padding = 1, dilation = 1, groups = 1, bias = True),
            nn.LeakyReLU(inplace = True))
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels = 16, out_channels = 16, kernel_size = 3, stride = 1, padding = 1, dilation = 1, groups = 1, bias = True),
            nn.LeakyReLU(inplace = True))
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = 3, stride = 2, padding = 1, dilation = 1, groups = 1, bias = True),
            nn.LeakyReLU(inplace = True))
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = 3, stride = 1, padding = 1, dilation = 1, groups = 1, bias = True),
            nn.LeakyReLU(inplace = True))
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 3, stride = 2, padding = 1, dilation = 1, groups = 1, bias = True),
            nn.LeakyReLU(inplace = True))
        self.conv6 = nn.Sequential(
            nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, stride = 1, padding = 1, dilation = 1, groups = 1, bias = True),
            nn.LeakyReLU(inplace = True))
        self.conv7 = nn.Sequential(
            nn.Conv2d(in_channels = 64, out_channels = 96, kernel_size = 3, stride = 2, padding = 1, dilation = 1, groups = 1, bias = True),
            nn.LeakyReLU(inplace = True))
        self.conv8 = nn.Sequential(
            nn.Conv2d(in_channels = 96, out_channels = 96, kernel_size = 3, stride = 1, padding = 1, dilation = 1, groups = 1, bias = True),
            nn.LeakyReLU(inplace = True))
        self.conv9 = nn.Sequential(
            nn.Conv2d(in_channels = 96, out_channels = 128, kernel_size = 3, stride = 2, padding = 1, dilation = 1, groups = 1, bias = True),
            nn.LeakyReLU(inplace = True))
        self.conv10 = nn.Sequential(
            nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = 3, stride = 1, padding = 1, dilation = 1, groups = 1, bias = True),
            nn.LeakyReLU(inplace = True))
        self.conv11 = nn.Sequential(
            nn.Conv2d(in_channels = 128, out_channels = 192, kernel_size = 3, stride = 2, padding = 1, dilation = 1, groups = 1, bias = True),
            nn.LeakyReLU(inplace = True))
        self.conv12 = nn.Sequential(
            nn.Conv2d(in_channels = 192, out_channels = 192, kernel_size = 3, stride = 1, padding = 1, dilation = 1, groups = 1, bias = True),
            nn.LeakyReLU(inplace = True))
    

    def forward(self, x):
        args = self.args
        out_conv1 = self.conv1(x)
        out_conv2 = self.conv2(out_conv1)
        out_conv3 = self.conv3(out_conv2)
        out_conv4 = self.conv4(out_conv3)
        out_conv5 = self.conv5(out_conv4)
        out_conv6 = self.conv6(out_conv5)
        out_conv7 = self.conv7(out_conv6)
        out_conv8 = self.conv8(out_conv7)
        out_conv9 = self.conv9(out_conv8)
        out_conv10 = self.conv10(out_conv9)
        out_conv11 = self.conv11(out_conv10)
        out_conv12 = self.conv12(out_conv11)

        # print(x.size(),
        # out_conv1.size(), out_conv2.size(), out_conv3.size(), out_conv4.size(),
        # out_conv5.size(), out_conv6.size(), out_conv7.size(), out_conv8.size(),
        # out_conv9.size(), out_conv10.size(), out_conv11.size(), out_conv12.size(),
        # sep = '\n')

        return out_conv2, out_conv4, out_conv6, out_conv8, out_conv10, out_conv12
        

class OpticalFlowEstimator(nn.Module):

    def __init__(self, args):
        super(OpticalFlowEstimator, self).__init__()
        self.args = args

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels = 115, out_channels = 128, kernel_size = 3, stride = 1, padding = 1, dilation = 1, groups = 1, bias = True),
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
        x = torch.cat([tgt, cost_volume, flow], dim = 1)
        
        out_conv1 = self.conv1(x)
        out_conv2 = self.conv2(out_conv1)
        out_conv3 = self.conv3(out_conv2)
        out_conv4 = self.conv4(out_conv3)
        out_feature = self.conv5(out_conv4)
        out_flow = self.conv6(out_feature)

        print(x.size(),
        out_conv1.size(), out_conv2.size(), out_conv3.size(), out_conv4.size(), 
        out_feature.size(), out_flow.size(),
        sep = '\n')
        
        return out_feature, out_flow

class ContextNetwork(nn.Module):


    def __init__(self, args):
        super(ContextNetwork, self).__init__()
        self.args = args
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels = 34, out_channels = 128, kernel_size = 3, stride = 1, padding = 1, dilation = 1, groups = 1, bias = True),
            nn.LeakyReLU(inplace = True))
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = 3, stride = 1, padding = 1, dilation = 2, groups = 1, bias = True),
            nn.LeakyReLU(inplace = True))
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = 3, stride = 1, padding = 1, dilation = 4, groups = 1, bias = True),
            nn.LeakyReLU(inplace = True))
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels = 128, out_channels = 96, kernel_size = 3, stride = 1, padding = 1, dilation = 8, groups = 1, bias = True),
            nn.LeakyReLU(inplace = True))
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels = 96, out_channels = 64, kernel_size = 3, stride = 1, padding = 1, dilation = 16, groups = 1, bias = True),
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

        print(x.size(),
        out_conv1.size(), out_conv2.size(), out_conv3.size(), out_conv4.size(),
        out_conv5.size(), out_conv6.size(),
        out_flow.size(),
        sep = '\n')


        return flow + out_flow