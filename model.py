import torch
from time import time
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from modules import (FeaturePyramidExtractor, CostVolumeLayer, OpticalFlowEstimator, ContextNetwork)
from correlation_package.modules.correlation import Correlation


class Net(nn.Module):


    def __init__(self, args):
        super(Net, self).__init__()
        self.args = args

        # build feature layer
        # ============================================================
        self.feature_pyramid_extractor = FeaturePyramidExtractor(args).to(args.device)


        # build corr layer
        # ============================================================
        if args.corr == 'cost_volume':
            self.corr = CostVolumeLayer(args).to(args.device)
        elif args.corr == 'flownetc':
            self.corr = Correlation(pad_size=20, kernel_size=1, max_displacement=20, stride1=1, stride2=2, corr_multiply=1).to(args.device)


        # build estimate layer
        # ============================================================
        if args.corr != 'none':
            self.optical_flow_estimators = []
            for layer_idx in range(args.num_levels):
                layer = OpticalFlowEstimator(args, args.lv_chs[layer_idx] + (args.search_range*2+1)**2 + 2).to(args.device)
                self.optical_flow_estimators.append(layer)
                self.add_module(f'FlowEstimator(Lv{layer_idx + 1})', layer)
        else:
            self.optical_flow_estimators = []
            for layer_idx in range(args.num_levels):
                layer = OpticalFlowEstimator(args, 2 * args.lv_chs[layer_idx] + 2).to(args.device)
                self.optical_flow_estimators.append(layer)
                self.add_module(f'FlowEstimator(Lv{layer_idx + 1})', layer)


        # build context layer
        # ============================================================
        if args.use_context_network:
            self.context_networks = []
            for layer_idx in range(args.num_levels):
                layer = ContextNetwork(args, args.lv_chs[layer_idx] + 2).to(args.device)
                self.context_networks.append(layer)
                self.add_module(f'ContextNetwork(Lv{layer_idx + 1})', layer)
        if args.use_warping_layer:
            self.grid_pyramid = None


    def forward(self, inputs):
        args = self.args
        src_img, tgt_img = inputs
        # t = time()
        src_features = self.feature_pyramid_extractor(src_img)
        tgt_features = self.feature_pyramid_extractor(tgt_img)
        # print(f'Extract Features of Sources: {time() - t: .2f}s'); t = time()
        # TypeError: Type torch.cuda.FloatTensor doesn't implement stateless method linspace
        # so making grids is done on CPU, and Tensors will be converted to cuda.Tensor and dispatch to GPUs
        # compute grid on each level
        if args.use_warping_layer:
            if self.grid_pyramid is None:
                self.grid_pyramid = []
                for layer_idx in range(args.num_levels):
                    x = src_features[layer_idx]
                    torchHorizontal = torch.linspace(-1.0, 1.0, x.size(3)).to(args.device).view(1, 1, 1, x.size(3)).expand(x.size(0), 1, x.size(2), x.size(3))
                    torchVertical = torch.linspace(-1.0, 1.0, x.size(2)).to(args.device).view(1, 1, x.size(2), 1).expand(x.size(0), 1, x.size(2), x.size(3))
                    grid = torch.cat([torchHorizontal, torchVertical], 1)
                    self.grid_pyramid.append(grid)
            grid_pyramid = self.grid_pyramid

        flow_features, flow_pyramid, flow_refined_pyramid = [], [], []
        B, C, H, W = src_features[-1].size()
        for layer_idx in range(args.num_levels - 1, -1, -1):
            # upsample the flow estimated from upper level
            flow = torch.zeros((B, 2, H, W)).to(args.device) if layer_idx == args.num_levels - 1 else F.upsample(flow, scale_factor = 2, mode = 'bilinear')
            # warp tgt_feature
            # print(tgt_features[l].size(), grid_pyramid[l].size(), flow.size())
            
            tgt_feature = tgt_features[layer_idx]
            if args.use_warping_layer:
                tgt_feature = F.grid_sample(tgt_feature, (grid_pyramid[layer_idx] + flow).permute(0, 2, 3, 1))
            
            # build cost volume, time costly
            if args.corr != 'none':
                corr = self.corr(src_features[layer_idx], tgt_feature)
                x = torch.cat([src_features[layer_idx], cost_volume, flow], dim = 1)
            else:
                x = torch.cat([src_features[layer_idx], tgt_feature, flow], dim = 1)
                
            flow = self.optical_flow_estimators[layer_idx](x)
            # use context to refine
            if args.use_context_network:
                flow = self.context_networks[layer_idx](src_features[layer_idx], flow)
            flow_pyramid.append(flow)

            # output
            if layer_idx == args.output_level or layer_idx == 0:
                flow = F.upsample(flow, scale_factor = 2**(layer_idx+1), mode = 'bilinear')
                break

        return flow, flow_pyramid