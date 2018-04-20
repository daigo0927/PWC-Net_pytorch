import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from modules import (FeaturePyramidExtractor, CostVolumeLayer, OpticalFlowEstimator, ContextNetwork)


class Net(nn.Module):


    def __init__(self, args):
        super(Net, self).__init__()
        self.args = args
        self.feature_pyramid_extractor = FeaturePyramidExtractor(args)
        self.cost_volume_layer = CostVolumeLayer(args)
        self.optical_flow_estimators = [OpticalFlowEstimator(args, ch_in + (args.search_range*2+1)**2 + 2) for ch_in in (192, 128, 96, 64, 32, 16)]
        self.context_networks = [ContextNetwork(args, ch_in + 2) for ch_in in (192, 128, 96, 64, 32, 16)]
    
    def cuda_(self):
        self.feature_pyramid_extractor.levels = [i.cuda() for i in self.feature_pyramid_extractor.levels]
        self.feature_pyramid_extractor.cuda()
        self.optical_flow_estimators = [i.cuda() for i in self.optical_flow_estimators]
        self.context_networks = [i.cuda() for i in self.context_networks]
        self.cuda()

        self.cost_volume_layer = nn.DataParallel(self.cost_volume_layer)
        self.optical_flow_estimators = [nn.DataParallel(i) for i in self.optical_flow_estimators]
        self.context_networks = [nn.DataParallel(i) for i in self.context_networks]



    def forward(self, src_img, tgt_img):
        args = self.args
        # (B,3,H,W) -> (B,3,H/2,W/2) -> (B,3,H/4,W/4) -> (B,3,H/8,W/8)
        src_features = self.feature_pyramid_extractor(src_img)
        tgt_features = self.feature_pyramid_extractor(tgt_img)

        # TypeError: Type torch.cuda.FloatTensor doesn't implement stateless method linspace
        # so making grids is done on CPU, and Tensors will be converted to cuda.Tensor and dispatch to GPUs
        # compute grid on each level
        grid_pyramid = []
        for l in range(args.num_levels):
            x = src_features[l]
            torchHorizontal = torch.linspace(-1.0, 1.0, x.size(3)).view(1, 1, 1, x.size(3)).expand(x.size(0), 1, x.size(2), x.size(3))
            torchVertical = torch.linspace(-1.0, 1.0, x.size(2)).view(1, 1, x.size(2), 1).expand(x.size(0), 1, x.size(2), x.size(3))
            grid = torch.cat([torchHorizontal, torchVertical], 1).cuda()
            grid = Variable(data = grid, volatile = not self.training)
            grid_pyramid.append(grid)

        
        flow_pyramid, flow_refined_pyramid = [], []
        flow_features = []
        for l in range(args.num_levels):
            # upsample the flow estimated from upper level
            flow = torch.zeros_like(src_features[0])[:,:2,:,:] if l == 0 else F.upsample(flow, scale_factor = 2, mode = 'bilinear')
            # warp tgt_feature
            tgt_feature_warped = F.grid_sample(tgt_features[l], (grid_pyramid[l] + flow).permute(0, 2, 3, 1))
            # build cost volume, time costly
            cost_volume = self.cost_volume_layer(src_features[l], tgt_feature_warped)
            # estimate flow
            flow_feature, flow = self.optical_flow_estimators[l](src_features[l], cost_volume, flow)

            # use context to refine
            flow_refined = self.context_networks[l](src_features[l], flow)

            flow_features.append(flow_feature)
            flow_pyramid.append(flow)
            flow_refined_pyramid.append(flow_refined)

        summaries = dict()
        summaries['flow_feature'] = flow_features
        summaries['coarse_flow_pyramid'] = flow_pyramid
        return flow_refined_pyramid, summaries