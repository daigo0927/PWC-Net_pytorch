import torch
from time import time
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from modules import (FeaturePyramidExtractor, CostVolumeLayer, OpticalFlowEstimator, ContextNetwork)


class Net(nn.Module):


    def __init__(self, args):
        super(Net, self).__init__()
        self.args = args
        device = torch.device(args.device)
        self.feature_pyramid_extractor = FeaturePyramidExtractor(args).to(device)
        if args.use_cost_volume_layer:
            self.cost_volume_layer = CostVolumeLayer(args).to(device)
            self.optical_flow_estimators = []
            for layer_idx in range(args.num_levels):
                layer = OpticalFlowEstimator(args, args.lv_chs[layer_idx] + (args.search_range*2+1)**2 + 2).to(device)
                self.optical_flow_estimators.append(layer)
                self.add_module(f'FlowEstimator(Lv{layer_idx + 1})', layer)
        else:
            self.optical_flow_estimators = []
            for layer_idx in range(args.num_levels):
                layer = OpticalFlowEstimator(args, 2 * args.lv_chs[layer_idx] + 2).to(device)
                self.optical_flow_estimators.append(layer)
                self.add_module(f'FlowEstimator(Lv{layer_idx + 1})', layer)
        if args.use_context_network:
            self.context_networks = []
            for layer_idx in range(args.num_levels):
                layer = ContextNetwork(args, args.lv_chs[layer_idx] + 2).to(device)
                self.context_networks.append(layer)
                self.add_module(f'ContextNetwork(Lv{layer_idx + 1})', layer)
        if args.use_warping_layer:
            self.grid_pyramid = None


    def forward(self, inputs):
        args = self.args
        device = torch.device(args.device)
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
                    torchHorizontal = torch.linspace(-1.0, 1.0, x.size(3)).to(device).view(1, 1, 1, x.size(3)).expand(x.size(0), 1, x.size(2), x.size(3))
                    torchVertical = torch.linspace(-1.0, 1.0, x.size(2)).to(device).view(1, 1, x.size(2), 1).expand(x.size(0), 1, x.size(2), x.size(3))
                    grid = torch.cat([torchHorizontal, torchVertical], 1)
                    self.grid_pyramid.append(grid)
            grid_pyramid = self.grid_pyramid
        # print(f'Build Grids: {time() - t: .2f}s'); t = time()
        


        flow_features, flow_pyramid, flow_refined_pyramid = [], [], []
        B, C, H, W = src_features[-1].size()
        for layer_idx in range(args.num_levels - 1, 0, -1):
            # upsample the flow estimated from upper level
            flow = torch.zeros((B, 2, H, W)).to(device) if layer_idx == args.num_levels - 1 else F.upsample(flow, scale_factor = 2, mode = 'bilinear')
            # warp tgt_feature
            # print(tgt_features[l].size(), grid_pyramid[l].size(), flow.size())
            
            tgt_feature = tgt_features[layer_idx]
            if args.use_warping_layer:
                tgt_feature = F.grid_sample(tgt_feature, (grid_pyramid[layer_idx] + flow).permute(0, 2, 3, 1))
            
            # build cost volume, time costly
            if args.use_cost_volume_layer:
                cost_volume = self.cost_volume_layer(src_features[layer_idx], tgt_feature)
                x = torch.cat([src_features[layer_idx], cost_volume, flow], dim = 1)
            else:
                x = torch.cat([src_features[layer_idx], tgt_feature, flow], dim = 1)
                
            flow = self.optical_flow_estimators[layer_idx](x)


            # use context to refine
            if args.use_context_network:
                flow = self.context_networks[layer_idx](src_features[layer_idx], flow)

            # output
            if layer_idx == args.output_level:
                output_flow = F.upsample(flow, scale_factor = 2**layer_idx, mode = 'bilinear')
                break
            # print(f'[Lv{l}] Refine Flow: {time() - t: .2f}s'); t = time()

        return output_flow, flow_pyramid