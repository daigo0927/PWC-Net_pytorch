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
        if args.no_cost_volume:
            self.optical_flow_estimators = [OpticalFlowEstimator(args, ch_in + ch_in + 2).to(device) for ch_in in args.lv_chs[::-1]]
        else:
            self.cost_volume_layer = CostVolumeLayer(args).to(device)
            self.optical_flow_estimators = [OpticalFlowEstimator(args, ch_in + (args.search_range*2+1)**2 + 2).to(device) for ch_in in args.lv_chs[::-1]]
        self.context_networks = [ContextNetwork(args, ch_in + 2).to(device) for ch_in in args.lv_chs[::-1]]
        self.grid_pyramid = None


    def forward(self, inputs):
        args = self.args
        device = torch.device(args.device)
        src_img, tgt_img = inputs
        # (B,3,H,W) -> (B,3,H/2,W/2) -> (B,3,H/4,W/4) -> (B,3,H/8,W/8)
        # t = time()
        src_features = self.feature_pyramid_extractor(src_img)
        # print(f'Extract Features of Sources: {time() - t: .2f}s'); t = time()
        tgt_features = self.feature_pyramid_extractor(tgt_img)
        # print(f'Extract Features of Sources: {time() - t: .2f}s'); t = time()
        # TypeError: Type torch.cuda.FloatTensor doesn't implement stateless method linspace
        # so making grids is done on CPU, and Tensors will be converted to cuda.Tensor and dispatch to GPUs
        # compute grid on each level
        if self.grid_pyramid is None:
            self.grid_pyramid = []
            for l in range(args.num_levels):
                x = src_features[l]
                torchHorizontal = torch.linspace(-1.0, 1.0, x.size(3)).view(1, 1, 1, x.size(3)).expand(x.size(0), 1, x.size(2), x.size(3)).to(device)
                torchVertical = torch.linspace(-1.0, 1.0, x.size(2)).view(1, 1, x.size(2), 1).expand(x.size(0), 1, x.size(2), x.size(3)).to(device)
                grid = torch.cat([torchHorizontal, torchVertical], 1)
                self.grid_pyramid.append(grid)
        grid_pyramid = self.grid_pyramid
        # print(f'Build Grids: {time() - t: .2f}s'); t = time()
        B, C, H, W = src_features[0].size()


        flow_features, flow_pyramid, flow_refined_pyramid = [], [], []
        for l in range(args.num_levels):
            # upsample the flow estimated from upper level
            if l > 0:
                flow = F.upsample(flow, scale_factor = 2, mode = 'bilinear')
            else:
                device = torch.device(args.device)
                flow = torch.zeros((B, 2, H, W)).to(device)
            # warp tgt_feature
            # print(tgt_features[l].size(), grid_pyramid[l].size(), flow.size())
            tgt_feature_warped = F.grid_sample(tgt_features[l], (grid_pyramid[l] + flow).permute(0, 2, 3, 1))
            # build cost volume, time costly
            if args.no_cost_volume:
                flow_feature, flow = self.optical_flow_estimators[l](src_features[l], tgt_feature_warped, flow)
                # print(f'[Lv{l}] Estimate Flow: {time() - t: .2f}s'); t = time()
            else:
                cost_volume = self.cost_volume_layer(src_features[l], tgt_feature_warped)
                # print(f'[Lv{l}] Compute Cost Volume: {time() - t: .2f}s'); t = time()
                # estimate flow
                flow_feature, flow = self.optical_flow_estimators[l](src_features[l], cost_volume, flow)
                # print(f'[Lv{l}] Estimate Flow: {time() - t: .2f}s'); t = time()

            # use context to refine
            flow_refined = self.context_networks[l](src_features[l], flow)
            # print(f'[Lv{l}] Refine Flow: {time() - t: .2f}s'); t = time()

            flow_features.append(flow_feature); flow_pyramid.append(flow); flow_refined_pyramid.append(flow_refined)
        

        for layer_idx, (w, flow, gt) in zip(args.weights, flow_pyramid, flow_gt_pyramid):
            print(layer_idx, torch.norm(flow - gt, p = 2, dim = 1).mean())

        summaries = dict()
        summaries['flow_feature'] = flow_features
        summaries['coarse_flow_pyramid'] = flow_pyramid
        return flow_refined_pyramid, summaries