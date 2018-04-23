import torch.nn.functional as F


def get_criterion(args):
    return training_loss if args.dataset in ('FlyingChairs', 'FlyingThings') else robust_training_loss


def training_loss(args, flow_pyramid, flow_gt_pyramid):
    return sum(args.weights[l] * (F.mse_loss(flow_pyramid[l], flow_gt_pyramid[l])) for l in range(args.num_levels))
    
    
def robust_training_loss(args, flow_pyramid, flow_gt_pyramid):
    return sum(args.weights[l] * ((flow_pyramid[l] - flow_gt_pyramid[l]).abs().sum() + args.epsilon) ** args.q for l in range(args.num_levels))
    