import torch
import torch.nn.functional as F


def get_criterion(args):
    return training_loss if args.loss == 'L2' else robust_training_loss

def L1loss(x, y): return (x - y).abs().mean()
def L2loss(x, y): return torch.norm(x - y, p = 2, dim = 1).mean()

def training_loss(args, flow_pyramid, flow_gt_pyramid):
    return sum(w * L2loss(flow, gt) for w, flow, gt in zip(args.weights, flow_pyramid, flow_gt_pyramid))
    
def robust_training_loss(args, flow_pyramid, flow_gt_pyramid):
    return sum((w * L1loss(flow - gt) + args.epsilon) ** args.q for w, flow, gt in zip(args.weights, flow_pyramid, flow_gt_pyramid))
    