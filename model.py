import torch.nn as nn
import nn.functional as F


from .modules import (FeaturePyramidExtractor, WarpingLayer, CostVolumeLayer, OpticalFlowEstimator, ContextNetwork)




class Net(nn.Module):


    def __init__(self):
        self.feature_pyramid_extractor = FeaturePyramidExtractor()
        self.warping_layer = WarpingLayer()
        self.cost_volume_layer = CostVolumeLayer()
        self.opticla_flow_estimator = OpticalFlowEstimator()
        self.context_network = ContextNetwork()
    

    def forward(self, src_img, tgt_img, flow_gt):
        src_features = self.feature_pyramid_extractor(src_img)
        tgt_features = self.feature_pyramid_extractor(tgt_img)

        # on each level:
        # 1. upsample the flow on upper level
        # 2. warp tgt_feature
        # 3. build cost volume
        # 4. estimate flow
        flow = Variable(torch.zeros_like(src_features[-1]))
        for l in range(args.num_levels):
            flow_upsampled = F.upsample_bilinear(flow, scale_factor = 2)
            tgt_feature_warped = self.warping_layer(tgt_features[l], flow_upsampled)
            flow_gt_downsampled = F.avg_pool2d(flow_gt, kernel_size = 2, stride = 2)

            cost_volume = self.cost_volume_layer(src_features[l], tgt_feature_warped)

            flow = self.opticla_flow_estimator(src_features[l], cost_volume)
            final_flow, flow_pyramids = 0,0
        
        return flow_pyramids