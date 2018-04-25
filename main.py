from datetime import datetime
import argparse
import imageio
import cv2
import numpy as np
import torch
from functools import partial
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
import time

from model import Net
from losses import get_criterion
from dataset import (FlyingChairs, FlyingThings, Sintel, SintelFinal, SintelClean, KITTI)

import tensorflow as tf
from summary import summary
from logger import Logger
from pathlib import Path
from flow_utils import (vis_flow, save_flow)


def main():
    parser = argparse.ArgumentParser(description='Structure from Motion Learner training on KITTI and CityScapes Dataset',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # mode selection
    # ============================================================
    modes = parser.add_subparsers(title='modes',  
                                description='valid modes',  
                                help='additional help',  
                                dest='subparser_name')
    train_parser = modes.add_parser('train'); train_parser.set_defaults(func = train)
    pred_parser = modes.add_parser('pred'); pred_parser.set_defaults(func = pred)
    test_parser = modes.add_parser('eval'); test_parser.set_defaults(func = test)



    # train_parser
    # ============================================================
    # dataflow
    train_parser.add_argument('--crop_type', type = str, default = 'random')
    train_parser.add_argument('--crop_shape', type = int, nargs = '+', default = [384, 448])
    train_parser.add_argument('--resize_shape', nargs = 2, type = int, default = None)
    train_parser.add_argument('--resize_scale', type = float, default = None)
    train_parser.add_argument('--num_workers', default = 8, type = int, help = 'num of workers')
    train_parser.add_argument('--batch_size', default = 8, type=int, help='mini-batch size')
    train_parser.add_argument('--dataset_dir', type = str)
    train_parser.add_argument('--dataset', type = str)

    # net
    train_parser.add_argument('--num_levels', type = int, default = 6)
    train_parser.add_argument('--lv_chs', nargs = '+', type = int, default = [16, 32, 64, 96, 128, 192])
    train_parser.add_argument('--no_cost_volume', action = 'store_true')

    # loss
    train_parser.add_argument('--weights', nargs = '+', type = float, default = [1,0.32,0.08,0.02,0.01,0.005])
    train_parser.add_argument('--epsilon', default = 0.02)
    train_parser.add_argument('--q', type = int, default = 0.4)
    train_parser.add_argument('--loss', type = str, default = 'L2')
    
    # optimize
    train_parser.add_argument('--lr', type = float, default = 4e-4)
    train_parser.add_argument('--momentum', default = 4e-4)
    train_parser.add_argument('--beta', default = 0.99)
    train_parser.add_argument('--weight_decay', default = 4e-4)
    train_parser.add_argument('--total_step', default = 200 * 1000)

    # summary & log args
    train_parser.add_argument('--log_dir', default = 'train_log/' + datetime.now().strftime('%Y%m%d-%H%M%S'))
    train_parser.add_argument('--summary_interval', type = int, default = 100)
    train_parser.add_argument('--log_interval', type = int, default = 100)
    train_parser.add_argument('--checkpoint_interval', type = int, default = 100)
    train_parser.add_argument('--max_output', type = int, default = 3)



    # pred_parser
    # ============================================================
    pred_parser.add_argument('-i', '--input', nargs = 2)
    pred_parser.add_argument('-o', '--output', default = 'output.flo')
    pred_parser.add_argument('--load', type = str)



    # eval_parser
    # ============================================================
    test_parser.add_argument('--load', type = str)



    # public_parser
    # ============================================================
    parser.add_argument('--search_range', type = int, default = 4)
    parser.add_argument('--no_cuda', action = 'store_true')

    args = parser.parse_args()


    # check args
    # ============================================================
    if args.subparser_name == 'train':
        assert len(args.weights) == len(args.lv_chs) == args.num_levels
        assert args.dataset in ['FlyingChairs', 'FlyingThings', 'SintelFinal', 'SintelClean', 'KITTI'], 'One dataset should be correctly set as for there are specific hyper-parameters for every dataset'
    elif args.subparser_name == 'pred':
        assert args.input is not None, 'TWO input image path should be given.'
        assert args.load is not None
    elif args.subparser_name == 'test':
        assert not(args.train or args.predict), 'Only ONE mode should be selected.'
        assert args.load is not None
    else:
        raise RuntimeError('use train/predict/test to select a mode')

    args.func(args)



def train(args):
    # Build Model
    # ============================================================
    model = nn.DataParallel(Net(args))
    if not args.no_cuda: model.cuda()
    # summary(model, input_size = [(3, 384, 448)] * 2)
    # quit()

    # TODO: change optimizer to S_long & S_fine (same as flownet2)
    
    # build criterion
    criterion = get_criterion(args)
    optimizer = torch.optim.Adam(model.modules()[0].parameters(), args.lr,
                                 betas = (args.momentum, args.beta),
                                 weight_decay = args.weight_decay)


    
    # Prepare Dataloader
    # ============================================================
    train_dataset, eval_dataset = eval("{0}('{1}', 'train', cropper = '{5}', crop_shape = {2}, resize_shape = {3}, resize_scale = {4}), {0}('{1}', 'test', cropper = '{5}', crop_shape = {2}, resize_shape = {3}, resize_scale = {4})".format(args.dataset, args.dataset_dir, args.crop_shape, args.resize_shape, args.resize_scale, args.crop_type))

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
    logger = Logger(args.log_dir)
    p_log = Path(args.log_dir)

    iter_time = 0

    # Start training
    # ============================================================
    data_iter = iter(train_loader)
    iter_per_epoch = len(train_loader)
    model.train()
    for step in range(1, args.total_step + 1):
        t_iter = time.time()

        # Reset the data_iter
        if (step) % iter_per_epoch == 0: data_iter = iter(train_loader)

        # Load Data
        # ============================================================
        data, target = next(data_iter)
        # shape: B,3,H,W
        squeezer = partial(torch.squeeze, dim = 2)
        src_img, tgt_img = map(squeezer, data[0].split(split_size = 1, dim = 2))
        # shape: B,2,H,W
        flow_gt = target[0]
        if not args.no_cuda: src_img, tgt_img, flow_gt = map(lambda x: x.cuda(), (src_img, tgt_img, flow_gt))
        src_img, tgt_img, flow_gt = map(Variable, (src_img, tgt_img, flow_gt))


        
        # Build Groundtruth Pyramid
        # ============================================================
        flow_gt_pyramid = []
        x = flow_gt
        for l in range(args.num_levels):
            x = F.avg_pool2d(x, 2)
            flow_gt_pyramid.insert(0, x)

        # Forward Pass
        # ============================================================
        # features on each level will downsample to 1/2 from bottom to top
        flow_pyramid, summaries = model([src_img, tgt_img])

        
        # Compute Loss
        # ============================================================
        loss = criterion(args, flow_pyramid, flow_gt_pyramid)


        
        # Do step
        # ============================================================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        iter_time += time.time() - t_iter

        
        # Collect Summaries & Output Logs
        # ============================================================
        # TODO: add summaries and check
        # flow output on each level
        if step % args.summary_interval == 0:
            # add scalar summaries
            logger.scalar_summary('loss', loss.data[0], step)
            if 'epe' in locals():
                logger.scalar_summary('EPE', epe, step)


            # add image summaries
            B = flow_pyramid[l].size(0)
            for l in range(args.num_levels):
                # build flow & gt vis_pair:
                flow_vis = [vis_flow(i.squeeze()) for i in np.split(np.array(flow_pyramid[l].data).transpose(0,2,3,1), B, axis = 0)][:min(B, args.max_output)]
                flow_gt_vis = [vis_flow(i.squeeze()) for i in np.split(np.array(flow_gt_pyramid[l].data).transpose(0,2,3,1), B, axis = 0)][:min(B, args.max_output)]

                logger.image_summary(f'flow&gt_level{l}', [np.concatenate([i,j], axis = 1) for i,j in zip(flow_vis, flow_gt_vis)], step)
            logger.image_summary('src & tgt', [np.concatenate([i.squeeze(0),j.squeeze(0)], axis = 1) for i,j in zip(np.split(np.array(src_img.data).transpose(0,2,3,1), B, axis = 0), np.split(np.array(tgt_img.data).transpose(0,2,3,1), B, axis = 0))], step)
            
            # list(map(lambda x: x.squeeze(0), np.split(np.array(src_img.data).transpose(0,2,3,1), B, axis = 0)))[:3], step)
            # logger.image_summary('tgt_img', list(map(lambda x: x.squeeze(0), np.split(np.array(tgt_img.data).transpose(0,2,3,1), B, axis = 0)))[:3], step)
                # logger.image_summary(f'')
        # save model
        if step % args.checkpoint_interval == 0:
            torch.save(model.state_dict(), str(p_log / f'{step}.pkl'))
        # print log
        if step % args.log_interval == 0:
            epe = torch.norm(flow_pyramid[-1] - flow_gt_pyramid[-1], p = 2, dim = 1).mean()
            print(f'Step [{step}/{args.total_step}], Loss: {loss.data[0]:.4f}, EPE: {epe.data[0]:.4f}, Average Iter Time: {iter_time/step} per iter')



def pred(args):
    # TODO
    # Build Model
    # ============================================================
    model = Net(args)
    if not args.no_cuda: model.cuda_()
    model.load_state_dict(torch.load(args.load))
    model.eval()
    
    # Load Data
    # ============================================================
    src_img, tgt_img = map(imageio.imread, args.input)

    class StaticCenterCrop(object):
        def __init__(self, image_size, crop_size):
            self.th, self.tw = crop_size
            self.h, self.w = image_size
            print(self.th, self.tw, self.h, self.w)
        def __call__(self, img):
            return img[(self.h-self.th)//2:(self.h+self.th)//2, (self.w-self.tw)//2:(self.w+self.tw)//2,:]

    src_img = np.array(src_img)
    tgt_img = np.array(tgt_img)

    if args.crop_shape is not None:
        cropper = StaticCenterCrop(src_img.shape[:2], args.crop_shape)
        src_img = cropper(src_img)
        tgt_img = cropper(tgt_img)
    if args.resize_shape is not None:
        resizer = partial(cv2.resize, dsize = (0,0), dst = args.resize_shape)
        src_img, tgt_img = map(resizer, [src_img, tgt_img])
    elif args.resize_scale is not None:
        resizer = partial(cv2.resize, dsize = (0,0), fx = args.resize_scale, fy = args.resize_scale)
        src_img, tgt_img = map(resizer, [src_img, tgt_img])

    src_img = src_img[np.newaxis,:,:,:].transpose(0,3,1,2)
    tgt_img = tgt_img[np.newaxis,:,:,:].transpose(0,3,1,2)


    src_img = Variable(torch.Tensor(src_img))
    tgt_img = Variable(torch.Tensor(tgt_img))
    if not args.no_cuda: src_img, tgt_img = map(lambda x: x.cuda(), [src_img, tgt_img])
    

    # Forward Pass
    # ============================================================
    flow_pyramid, summaries = model(src_img, tgt_img)
    flow = flow_pyramid[-1]
    flow = np.array(flow.data).transpose(0,2,3,1).squeeze(0)
    save_flow(args.output, flow)
    flow_vis = vis_flow(flow)
    imageio.imwrite(args.output.replace('.flo', '.png'), flow_vis)
    import matplotlib.pyplot as plt
    plt.imshow(flow_vis)
    plt.show()



def test(args, eval_iter, ):
    # TODO
    pass



if __name__ == '__main__':
    main()