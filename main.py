from datetime import datetime
import argparse
import imageio

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader

from model import Net
from losses import get_criterion
from dataset import (FlyingChairs, FlyingThings, Sintel, KITTI)
try:
    import tensorflow as tf
    use_logger = True
except Exception:
    use_logger = False

if use_logger:
    from logger import Logger
from flow_utils import (flow_to_image, save_flow)


def parse():
    parser = argparse.ArgumentParser(description='Structure from Motion Learner training on KITTI and CityScapes Dataset',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # mode selection
    # ============================================================
    parser.add_argument('--train', action = 'store_true')
    parser.add_argument('--predict', action = 'store_true')
    parser.add_argument('--test', action = 'store_true')



    # mode=train args
    # ============================================================
    
    parser.add_argument('--log_dir', default = 'train_log/' + datetime.now().strftime('%Y%m%d-%H%M%S'))
    parser.add_argument('--dataset_dir', type = str)
    parser.add_argument('--dataset', type = str)
    parser.add_argument('--weights', nargs = '+', default = [0.32, 0.08, 0.02, 0.01, 0.005])
    parser.add_argument('--epsilon', default = 0.02)
    parser.add_argument('--q', default = 0.4)
    parser.add_argument('--gamma', default = 4e-4)
    parser.add_argument('--lr', default = 4e-4)
    parser.add_argument('--momentum', default = 4e-4)
    parser.add_argument('--beta', default = 0.99)
    parser.add_argument('--weight_decay', default = 4e-4)
    parser.add_argument('--total_step', default = 200 * 1000)
    # summary & log args
    parser.add_argument('--summary_interval', type = int, default = 100)
    parser.add_argument('--log_interval', type = int, default = 100)



    # mode=predict args
    # ============================================================
    parser.add_argument('-i', '--input', nargs = 2)
    parser.add_argument('-o', '--output', default = 'output.flo')



    # shared args
    # ============================================================
    parser.add_argument('--search_range', type = int, default = 4)
    parser.add_argument('--model', type = str)
    parser.add_argument('--no_cuda', action = 'store_true')
    parser.add_argument('--num_workers', default = 1, type = int, help = 'num of workers')
    parser.add_argument('-b', '--batch-size', default = 8, type=int, help='mini-batch size')


    # image input size
    # ============================================================
    parser.add_argument('--crop_shape', type = int, nargs = '+', default = [512, 512])
    parser.add_argument('--num_levels', type = int, default = 6)

    
    

    args = parser.parse_args()
    # check args
    # ============================================================
    if args.train:
        assert not(args.predict or args.test), 'Only ONE mode should be selected.'
        assert args.dataset in ['FlyingChairs', 'FlyingThings', 'Sintel', 'KITTI'], 'One dataset should be correctly set as for there are specific hyper-parameters for every dataset'
    elif args.predict:
        assert not(args.train or args.test), 'Only ONE mode should be selected.'
        assert args.input is not None, 'TWO input image path should be given.'
        assert args.model is not None
    elif args.test:
        assert not(args.train or args.predict), 'Only ONE mode should be selected.'
        assert args.model is not None
    else:
        raise RuntimeError('use --train/predict/test to select a mode')

    return args


def train(args):
    # Build Model
    # ============================================================
    
    model = Net(args)
    model.train()

    # TODO: change optimizer to S_long & S_fine (same as flownet2)
    
    # build criterion
    criterion = get_criterion(args)
    parameters = model.parameters()
    optimizer = torch.optim.Adam(parameters, args.lr,
                                 betas=(args.momentum, args.beta),
                                 weight_decay=args.weight_decay)


    
    # Prepare Dataloader
    # ============================================================
    train_dataset, eval_dataset = eval("{0}('data_train.txt'), {0}('data_test.txt')".format(args.dataset))

    train_loader = DataLoader(train_dataset,
                            batch_size = args.batch_size,
                            shuffle = True,
                            num_workers = args.num_workers,
                            pin_memory = True)
    eval_loader = DataLoader(eval_dataset,
                            batch_size = args.batch_size,
                            shuffle = True,
                            num_workers = args.num_workers,
                            pin_memory = True)



    # Init logger
    # ============================================================
    
    logger = Logger(args.log_dir)

    train_iter = iter(train_loader)

    for batch_idx, (data, target) in enumerate(train_iter):

        
        # Load Data
        # ============================================================
        # shape: B,3,H,W
        src_img, tgt_img = map(torch.squeeze, data[0].split(split_size = 1, dim = 2))
        # shape: B,2,H,W
        flow = target[0]
        
        if not args.no_cuda: src_img, tgt_img, flow = map(lambda x: x.cuda(), (src_img, tgt_img, flow))


        src_img, tgt_img, flow = map(Variable, (src_img, tgt_img, flow))
        
        # Forward Pass
        # ============================================================
        # features on each level will downsample to 1/2 from bottom to top
        (flow_pyramid, flow_gt_pyramid), summaries = model(src_img, tgt_img, flow)


        
        # Compute Loss
        # ============================================================
        loss = criterion(args, flow_pyramid, flow_gt_pyramid, model_parameters)


        
        # Do step
        # ============================================================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        
        # Collect Summaries & Output Logs
        # ============================================================
        # TODO: add summaries and check
        # flow output on each level
        if use_logger:
            if step % args.summary_interval == 0:
                # add scalar summaries
                logger.scalar_summary('loss', loss, step)
                logger.scalar_summary('EPE', epe, step)


                # add image summaries
                for l in args.num_levels:
                    logger.image_summary(f'flow_level{l}', [flow_pyramid[l]], step)
                    logger.image_summary(f'warped_level{l}', [warped_pyramid[l]], step)
                    # logger.image_summary(f'')
                pass
        
        if step % args.log_interval == 0:
            print(f'Step [{step}/{args.total_step}], Loss: {loss:.4f}, EPE: {epe:.4f}')


def predict(args):
    # TODO
    # Build Model
    # ============================================================
    model = Net(args)
    model.load_state_dict(torch.load(args.load))
    model.eval()
    
    
    src_img, tgt_img = map(imageio.imread, args.input)
    
    

    # Forward Pass
    # ============================================================
    flow_pyramid = model(src_img, tgt_img)
    flow = flow_pyramid[-1]
    save_flow(args.output, flow)
    flow_vis = flow_to_image(flow)
    imageio.imwrite(args.output.replace('.flo', '.png'), flow_vis)



def test(args):
    # TODO
    pass


def main(args):
    if args.train: train(args)
    elif args.predict: predict(args)
    else: test(args)


if __name__ == '__main__':
    args = parse()
    main(args)