import torch
from time import time
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from modules import (FeaturePyramidExtractor, CostVolumeLayer, OpticalFlowEstimator, ContextNetwork)
from correlation_package.modules.correlation import Correlation

from utils import get_grid



class Net(nn.Module):


    def __init__(self, args):
        super(Net, self).__init__()
        self.args = args

        self.feature_pyramid_extractor = FeaturePyramidExtractor(args).to(args.device)
        self.corr = Correlation(pad_size = args.search_range * 2 + 1, kernel_size = 1, max_displacement = args.search_range * 2 + 1, stride1 = 1, stride2 = 2, corr_multiply = 1).to(args.device)
        self.optical_flow_estimators = []
        for l in range(args.num_levels):
            layer = OpticalFlowEstimator(args, args.lv_chs[l] + (args.search_range*2+1)**2 + 2).to(args.device)
            self.add_module(f'FlowEstimator(Lv{l + 1})', layer)
            self.optical_flow_estimators.append(layer)
        self.context_networks = []
        for l in range(args.num_levels):
            layer = ContextNetwork(args, args.lv_chs[l] + 2).to(args.device)
            self.add_module(f'ContextNetwork(Lv{l + 1})', layer)
            self.context_networks.append(layer)


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
    
            flow *= 5

            # warp
            grid = (get_grid(x1).to(args.device) + flow).permute(0, 2, 3, 1)
            x2_warp = F.grid_samples(x2, grid)
            
            # concat and estimate flow
            flow_coarse = self.flow_estimators[l](torch.cat([x1, x2_warp, flow], dim = 1))

            # use context to refine the flow
            flow_fine = self.context_networks[l](torch.cat([x1, flow_coarse], dim = 1))
            flow = flow_coarse + flow_fine

            # collect
            flows.append(flow)
            summaries['x2_warps'].append(x2_warp)

        return flows, summaries


class NetOld(nn.Module):


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
            self.corr = Correlation(pad_size = args.search_range * 2 + 1, kernel_size = 1, max_displacement = args.search_range * 2 + 1, stride1 = 1, stride2 = 2, corr_multiply = 1).to(args.device)


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
        '''
        # outputs
        flows = []

        # tensors for summary
        summaries = {
            'x2_warps': [],

        }

        for l, (x1, x2) in enumerate(x1_pyramid, x2_pyramid):
            # upsample flow and scale the displacement
            shape = x1.size(); shape[1] = 2
            flow = torch.zeros(shape).to(args.device) if layer_idx == args.num_levels - 1 else F.upsample(flow, scale_factor = 2, mode = 'bilinear')
            flow *= 4

            # warp
            grid = (get_grid(x1) + flow).permute(0, 2, 3, 1)
            x2_warp = F.grid_samples(x2, grid)
            
            # concat and estimate flow
            flow_coarse = self.flow_estimators[l](torch.cat([x1, x2_warp, flow], dim = 1))

            # use context to refine the flow
            flow_fine = self.context_networks[l](torch.cat([x1, flow_coarse]. dim = 1))
            flow = flow_coarse + flow_fine

            # collect
            flows.append(flow)
            summaries['x2_warps'].append(x2_warp)

        return 
        '''
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
                x = torch.cat([src_features[layer_idx], corr, flow], dim = 1)
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