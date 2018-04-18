import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from modules import (FeaturePyramidExtractor, WarpingLayer, CostVolumeLayer, OpticalFlowEstimator, ContextNetwork)




class Net(nn.Module):


    def __init__(self, args):
        super(Net, self).__init__()
        self.args = args
        self.feature_pyramid_extractor = FeaturePyramidExtractor(args)
        self.warping_layer = WarpingLayer(args)
        self.cost_volume_layer = CostVolumeLayer(args)
        self.optical_flow_estimators = [OpticalFlowEstimator(args, ch_in + (args.search_range*2+1)**2 + 2) for ch_in in (192, 128, 96, 64, 32, 16)]
        self.context_networks = [ContextNetwork(args, ch_in + 2) for ch_in in (192, 128, 96, 64, 32, 16)]
    

    def forward(self, src_img, tgt_img):
        args = self.args
        # (B,3,H,W) -> (B,3,H/2,W/2) -> (B,3,H/4,W/4) -> (B,3,H/8,W/8)
        import time

        t = time.time()
        src_features = self.feature_pyramid_extractor(src_img)
        print(f'extract multi-level features from source: {time.time() - t}')

        t = time.time()
        tgt_features = self.feature_pyramid_extractor(tgt_img)
        print(f'extract multi-level features from target: {time.time() - t}')

        # on each level:
        # 1. upsample the flow on upper level
        # 2. warp tgt_feature
        # 3. build cost volume
        # 4. estimate flow
        flow_pyramid, flow_refined_pyramid = [], []
        flow_features = []
        for l in range(args.num_levels):
            flow = torch.zeros_like(src_features[0])[:,:2,:,:] if l == 0 else F.upsample(flow, scale_factor = 2, mode = 'bilinear')


            t = time.time()
            tgt_feature_warped = self.warping_layer(tgt_features[l], flow)
            print(f'[Lv {l}] warp target feature: {time.time() - t}')

            t = time.time()
            cost_volume = self.cost_volume_layer(src_features[l], tgt_feature_warped)
            print(f'[Lv {l}] compute cost volume: {time.time() - t}')

            t = time.time()
            flow_feature, flow = self.optical_flow_estimators[l](src_features[l], cost_volume, flow)
            print(f'[Lv {l}] estimate flow: {time.time() - t}')

            # use context to refine
            t = time.time()
            flow_refined = self.context_networks[l](src_features[l], flow)
            print(f'[Lv {l}] refine flow: {time.time() - t}')

            flow_features.append(flow_feature)
            flow_pyramid.append(flow)
            flow_refined_pyramid.append(flow_refined)

        summaries = dict()
        summaries['flow_feature'] = flow_features
        summaries['coarse_flow_pyramid'] = flow_pyramid
        return flow_refined_pyramid, summaries