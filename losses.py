import torch.nn.functional as F


def get_criterion(args):
    return training_loss if args.dataset in ('FlyingChairs', 'FlyingThings') else robust_training_loss


def training_loss(args, flow_pyramid, flow_gt_pyramid, model_parameters):
    loss = args.gamma * F.mse_loss(model_parameters, 0)
    for l in range(args.num_levels):
        loss += args.weights[l] * (F.mse_loss(flow_pyramid[l], flow_gt_pyramid[l]))
    
    return loss


def robust_training_loss(args, flow_pyramid, flow_gt_pyramid, model_parameters):
    loss = args.gamma * F.mse_loss(model_parameters, 0)
    for l in range(args.num_levels):
        loss += args.weights[l] * ((flow_pyramid[l] - flow_gt_pyramid[l]).abs() + args.epsilon) ** args.q
    
    return loss