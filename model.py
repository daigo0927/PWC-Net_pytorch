import torch
from time import time
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from modules import (WarpingLayer, FeaturePyramidExtractor, CostVolumeLayer, OpticalFlowEstimator, ContextNetwork)
from correlation_package.modules.correlation import Correlation


class Net(nn.Module):


    def __init__(self, args):
        super(Net, self).__init__()
        self.args = args

        self.feature_pyramid_extractor = FeaturePyramidExtractor(args).to(args.device)        
        
        self.warping_layer = WarpingLayer(args)
        if args.corr == 'CostVolumeLayer':
            self.corr = CostVolumeLayer(args)
        else:
            self.corr = Correlation(pad_size = args.search_range * 2 + 1, kernel_size = 1, max_displacement = args.search_range * 2 + 1, stride1 = 1, stride2 = 2, corr_multiply = 1).to(args.device)
        
        self.flow_estimators = []
        for l, ch in enumerate(args.lv_chs[::-1]):
            layer = OpticalFlowEstimator(args, ch + (args.search_range*2+1)**2 + 2).to(args.device)
            self.add_module(f'FlowEstimator(Lv{l})', layer)
            self.flow_estimators.append(layer)

        self.context_networks = []
        for l, ch in enumerate(args.lv_chs[::-1]):
            layer = ContextNetwork(args, ch + 2).to(args.device)
            self.add_module(f'ContextNetwork(Lv{l})', layer)
            self.context_networks.append(layer)

        # init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.bias is not None: nn.init.uniform_(m.bias)
                nn.init.xavier_uniform_(m.weight)

            if isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None: nn.init.uniform_(m.bias)
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        args = self.args

        if args.input_norm:
            rgb_mean = x.contiguous().view(x.size()[:2]+(-1,)).mean(dim=-1).view(x.size()[:2] + (1,1,1,))
            x = (x - rgb_mean) / args.rgb_max
        
        x1_raw = x[:,:,0,:,:]
        x2_raw = x[:,:,1,:,:]

        x1_pyramid = self.feature_pyramid_extractor(x1_raw)
        x2_pyramid = self.feature_pyramid_extractor(x2_raw)


        # outputs
        flows = []

        # tensors for summary
        summaries = {
            'x2_warps': [],

        }

        for l, (x1, x2) in enumerate(zip(x1_pyramid, x2_pyramid)):
            # upsample flow and scale the displacement
            if l == 0:
                shape = list(x1.size()); shape[1] = 2
                flow = torch.zeros(shape).to(args.device)
            else:
                flow = F.upsample(flow, scale_factor = 2, mode = 'bilinear')

            # warp
            x2_warp = self.warping_layer(x2, flow)
            
            # correlation
            corr = self.corr(x1, x2_warp)
            if args.corr_activation: F.leaky_relu_(corr)

            # concat and estimate flow
            # ATTENTION: `+ flow`` makes flow estimator learn to estimate residual flow
            flow_coarse = self.flow_estimators[l](torch.cat([x1, corr, flow], dim = 1)) + flow

            # use context to refine the flow
            flow_fine = self.context_networks[l](torch.cat([x1, flow_coarse], dim = 1))
            flow = flow_coarse + flow_fine

            # collect
            flows.append(flow)
            summaries['x2_warps'].append(x2_warp)

        return flows, summaries