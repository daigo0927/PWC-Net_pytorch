import torch
import torch.nn.functional as F


def get_criterion(args):
    return training_loss if args.loss == 'L2' else robust_training_loss


def training_loss(args, flow_pyramid, flow_gt_pyramid):
    return sum(w * (torch.norm(flow - gt, p = 2, dim = 1).mean()) for w, flow, gt in zip(args.weights, flow_pyramid, flow_gt_pyramid))
    
def robust_training_loss(args, flow_pyramid, flow_gt_pyramid):
    return sum(w * ((flow - gt).abs().mean() + args.epsilon) ** args.q for w, flow, gt in zip(args.weights, flow_pyramid, flow_gt_pyramid))
    