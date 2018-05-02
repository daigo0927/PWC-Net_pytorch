import torch
import torch.nn as nn
import torch.nn.functional as F

def L1loss(x, y): return (x - y).abs().mean()
def L2loss(x, y): return torch.norm(x - y, p = 2, dim = 1).mean()

def training_loss(args, flow_pyramid, flow_gt_pyramid):
    return sum(w * L2loss(flow, gt) for w, flow, gt in zip(args.weights, flow_pyramid, flow_gt_pyramid))
    
def robust_training_loss(args, flow_pyramid, flow_gt_pyramid):
    return sum((w * L1loss(flow, gt) + args.epsilon) ** args.q for w, flow, gt in zip(args.weights, flow_pyramid, flow_gt_pyramid))
    


def EPE(input_flow, target_flow):
    return torch.norm(target_flow-input_flow,p=2,dim=1).mean()


class L1(nn.Module):
    def __init__(self):
        super(L1, self).__init__()
    def forward(self, output, target):
        lossvalue = torch.abs(output - target).mean()
        return lossvalue


class L2(nn.Module):
    def __init__(self):
        super(L2, self).__init__()
    def forward(self, output, target):
        lossvalue = torch.norm(output-target,p=2,dim=1).mean()
        return lossvalue


class L1Loss(nn.Module):
    def __init__(self, args):
        super(L1Loss, self).__init__()
        self.args = args
        self.loss = L1()
        self.loss_labels = ['L1', 'EPE']

    def forward(self, output, target):
        lossvalue = self.loss(output, target)
        epevalue = EPE(output, target)
        return [lossvalue, epevalue]


class L2Loss(nn.Module):
    def __init__(self, args):
        super(L2Loss, self).__init__()
        self.args = args
        self.loss = L2()
        self.loss_labels = ['L2', 'EPE']

    def forward(self, output, target):
        lossvalue = self.loss(output, target)
        epevalue = EPE(output, target)
        return [lossvalue, epevalue]


class MultiScale(nn.Module):
    def __init__(self, args, startScale = 4, numScales = 6, l_weight= 0.32, norm= 'L1'):
        super(MultiScale,self).__init__()

        self.startScale = startScale
        self.numScales = numScales
        self.loss_weights = torch.FloatTensor([(l_weight / 2 ** scale) for scale in range(self.numScales)]).to(args.device)
        self.args = args
        self.l_type = norm
        self.div_flow = 0.05
        assert(len(self.loss_weights) == self.numScales)

        if self.l_type == 'L1':
            self.loss = L1()
        else:
            self.loss = L2()

        self.multiScales = [nn.AvgPool2d(self.startScale * (2**scale), self.startScale * (2**scale)) for scale in range(self.numScales)][::-1]
        self.loss_labels = ['MultiScale-'+self.l_type, 'EPE'],

    def forward(self, output, target):
        lossvalue = 0
        epevalue = 0

        if type(output) in (tuple, list):
            target = self.div_flow * target
            print(target.size())
            for i, output_ in enumerate(output):
                target_ = self.multiScales[i](target)
                print(target_.size(), output_.size())
                # epevalue += self.loss_weights[i]*EPE(output_, target_)
                # lossvalue += self.loss_weights[i]*self.loss(output_, target_)
            return [lossvalue, epevalue]
        else:
            epevalue += EPE(output, target)
            lossvalue += self.loss(output, target)
            return  [lossvalue, epevalue]

