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
from losses import L1loss, L2loss, training_loss, robust_training_loss, MultiScale
from dataset import (FlyingChairs, FlyingThings, Sintel, SintelFinal, SintelClean, KITTI)

import tensorflow as tf
from summary import summary as summary_
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

    summary_parser = modes.add_parser('summary'); summary_parser.set_defaults(func = summary)
    train_parser = modes.add_parser('train'); train_parser.set_defaults(func = train)
    pred_parser = modes.add_parser('pred'); pred_parser.set_defaults(func = pred)
    test_parser = modes.add_parser('eval'); test_parser.set_defaults(func = test)

    # public_parser
    # ============================================================
    parser.add_argument('--search_range', type = int, default = 4)
    parser.add_argument('--device', type = str, default = 'cuda')
    parser.add_argument('--rgb_max', type = float, default = 255)
    parser.add_argument('--residual', action = 'store_true')
    parser.add_argument('--flow_norm', action = 'store_true')


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
    train_parser.add_argument('--output_level', type = int, default = 4)
    train_parser.add_argument('--input_norm', action = 'store_true')
    train_parser.add_argument('--corr', type = str, default = 'cost_volume')

    # net
    train_parser.add_argument('--num_levels', type = int, default = 7)
    train_parser.add_argument('--lv_chs', nargs = '+', type = int, default = [16, 32, 64, 96, 128, 192])
    train_parser.add_argument('--corr_activation', action = 'store_true')
    train_parser.add_argument('--use_context_network', action = 'store_true')
    train_parser.add_argument('--use_warping_layer', action = 'store_true')
    train_parser.add_argument('--batch_norm', action = 'store_true')

    # loss
    train_parser.add_argument('--weights', nargs = '+', type = float, default = [0.32,0.08,0.02,0.01,0.005])
    train_parser.add_argument('--epsilon', default = 0.02)
    train_parser.add_argument('--q', type = int, default = 0.4)
    train_parser.add_argument('--loss', type = str, default = 'MultiScale')
    train_parser.add_argument('--optimizer', type = str, default = 'Adam')
    
    # optimize
    train_parser.add_argument('--lr', type = float, default = 1e-4)
    train_parser.add_argument('--momentum', default = 4e-4)
    train_parser.add_argument('--beta', default = 0.99)
    train_parser.add_argument('--weight_decay', type = float, default = 4e-4)
    train_parser.add_argument('--total_step', type = int, default = 200 * 1000)

    # summary & log args
    train_parser.add_argument('--log_dir', default = 'train_log/' + datetime.now().strftime('%Y%m%d-%H%M%S'))
    train_parser.add_argument('--summary_interval', type = int, default = 100)
    train_parser.add_argument('--log_interval', type = int, default = 100)
    train_parser.add_argument('--checkpoint_interval', type = int, default = 100)
    train_parser.add_argument('--gif_input', type = str, default = None)
    train_parser.add_argument('--gif_output', type = str, default = 'gif')
    train_parser.add_argument('--gif_interval', type = int, default = 100)
    train_parser.add_argument('--max_output', type = int, default = 3)



    # pred_parser
    # ============================================================
    pred_parser.add_argument('-i', '--input', nargs = 2)
    pred_parser.add_argument('-o', '--output', default = 'output.flo')
    pred_parser.add_argument('--load', type = str)



    # eval_parser
    # ============================================================
    test_parser.add_argument('--load', type = str)

    args = parser.parse_args()


    # check args
    # ============================================================
    if args.subparser_name == 'train':
        assert len(args.weights) >= args.output_level + 1
        assert len(args.lv_chs) + 1 == args.num_levels
        assert args.dataset in ['FlyingChairs', 'FlyingThings', 'SintelFinal', 'SintelClean', 'KITTI'], 'One dataset should be correctly set as for there are specific hyper-parameters for every dataset'
    elif args.subparser_name == 'pred':
        assert args.input is not None, 'TWO input image path should be given.'
        assert args.load is not None
    elif args.subparser_name == 'test':
        assert not(args.train or args.predict), 'Only ONE mode should be selected.'
        assert args.load is not None
    else:
        raise RuntimeError('use train/predict/test to select a mode')
    
    args.device = torch.device(args.device)

    args.func(args)


def summary(args):
    model = Net(args).to(args.device)
    summary_(model, (3, 384, 448))


def train(args):
    # Build Model
    # ============================================================
    model = Net(args).to(args.device)

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

    forward_time = 0
    backward_time = 0

    # Start training
    # ============================================================
    data_iter = iter(train_loader)
    iter_per_epoch = len(train_loader)
    Crit = eval(args.loss)
    criterion = MultiScale(args)


    # build criterion
    Opt = eval('torch.optim.' + args.optimizer)
    optimizer = Opt(model.parameters(), args.lr, weight_decay = args.weight_decay)

    total_loss = 0
    total_epe = 0
    total_loss_levels = [0] * args.num_levels
    total_epe_levels = [0] * args.num_levels
    # training
    # ============================================================
    for step in range(1, args.total_step + 1):
        # Reset the data_iter
        if (step) % iter_per_epoch == 0: data_iter = iter(train_loader)

        # Load Data
        # ============================================================
        data, target = next(data_iter)

        # shape: B,3,H,W
        squeezer = partial(torch.squeeze, dim = 2)
        # shape: B,2,H,W
        data, target = [d.to(args.device) for d in data], [t.to(args.device) for t in target]
        
        x1_raw, x2_raw = map(squeezer, data[0].split(split_size = 1, dim = 2))
        if x1_raw.size(0) != args.batch_size: continue
        flow_gt = target[0]


        # Forward Pass
        # ============================================================
        t_forward = time.time()
        flows, summaries = model(data[0])
        forward_time += time.time() - t_forward

        
        # Compute Loss
        # ============================================================
        loss, epe, loss_levels, epe_levels = criterion(flows, flow_gt)
        total_loss += loss.item()
        total_epe += epe.item()
        for l, (loss_, epe_) in enumerate(zip(loss_levels, epe_levels)):
            total_loss_levels[l] += loss_.item()
            total_epe_levels[l] += epe_.item()

        # if args.loss == 'L1':
        #     loss = L1loss(flow_gt, output_flow)
        # elif args.loss == 'PyramidL1':
        #     loss = robust_training_loss(args, flows, flow_gt_pyramid)
        # elif args.loss == 'L2':
        #     loss = L2loss(flow_gt, output_flow)
        # elif args.loss == 'PyramidL2':
        #     loss = training_loss(args, flows, flow_gt_pyramid)

        
        # backward
        # ============================================================
        t_backward = time.time()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        backward_time += time.time() - t_backward
        
        # Collect Summaries & Output Logs
        # ============================================================
        if step % args.summary_interval == 0:
            # Scalar Summaries
            # ============================================================
            logger.scalar_summary('lr', optimizer.param_groups[0]['lr'], step)
            logger.scalar_summary('loss', total_loss / step, step)
            logger.scalar_summary('EPE', total_epe / step, step)

            for l, (loss_, epe_) in enumerate(zip(loss_levels, epe_levels)):
                logger.scalar_summary(f'loss_lv{l}', total_loss_levels[l] / step, step)
                logger.scalar_summary(f'EPE_lv{l}', total_epe_levels[l] / step, step)

            # Image Summaries
            # ============================================================
            B = flows[0].size(0)
            for b in range(B):
                batch = [np.array(F.upsample(flows[l][b].unsqueeze(0), scale_factor = 2 ** ((len(flows)-l + 1))).detach().squeeze(0)).transpose(1,2,0) for l in range(len(flows) - 1)]
                # for i in batch:
                #     print(i.shape)
                # print(flows[-1][b].detach().cpu().numpy().transpose(1,2,0))
                # print(flow_gt[b].detach().cpu().numpy().transpose(1,2,0).shape)
                vis = np.concatenate(list(map(vis_flow, batch + [flows[-1][b].detach().cpu().numpy().transpose(1,2,0), flow_gt[b].detach().cpu().numpy().transpose(1,2,0)])), axis = 1)
                logger.image_summary(f'flow{b}', [vis.transpose(2, 0, 1)], step)
            
            
            
            # for l, x2_warp in enumerate(summaries['x2_warps']):
            #     out = [i.squeeze(0) for i in np.split(np.array(x2_warp.data).transpose(0,2,3,1), B, axis = 0)]
            #     for i in out:
            #         print(i.shape)
            #     logger.image_summary('tgt_warp', [i.squeeze(0) for i in np.split(np.array(x2_warp.data).transpose(0,2,3,1), B, axis = 0)], step)
            

            # for l, flow in enumerate(flows):
            #     flow_batchs[0], flow_batchs[1], flow_batchs[2] = [vis_flow(i.squeeze()) for i in np.split(np.array(F.upsample(flow, 2 ** (6-l)).transpose(0,2,3,1)), B, axis = 0)]
            
            
            # flow_vis = [vis_flow(i.squeeze()) for flow in flows for i in np.split(np.array(flow.data).transpose(0,2,3,1), B, axis = 0)][:min(B, args.max_output)]
            # for layer_idx, flow in enumerate(flows):
            #     flow_vis = 
            #     # flow_gt_vis = [vis_flow(i.squeeze()) for i in np.split(np.array(flow_gt_pyramid[layer_idx].data).transpose(0,2,3,1), B, axis = 0)][:min(B, args.max_output)]
            #     logger.image_summary(f'flow-lv{layer_idx}', flow_vis, step)

            logger.image_summary('src & tgt', [np.concatenate([i.squeeze(0),j.squeeze(0)], axis = 1) for i,j in zip(np.split(np.array(x1_raw.data).transpose(0,2,3,1), B, axis = 0), np.split(np.array(x2_raw.data).transpose(0,2,3,1), B, axis = 0))], step)

        # save model
        if step % args.checkpoint_interval == 0:
            torch.save(model.state_dict(), str(p_log / f'{step}.pkl'))
        # print log
        if step % args.log_interval == 0:
            print(f'Step [{step}/{args.total_step}], Loss: {total_loss / step:.4f}, EPE: {total_epe / step:.4f}, Forward: {forward_time/step*1000} ms, Backward: {backward_time/step*1000} ms')
        
        if step % args.gif_interval == 0:
            ...



def pred(args):
    # Get environment
    # Build Model
    # ============================================================
    model = Net(args).to(args.device)
    model.load_state_dict(torch.load(args.load))
    
    # Load Data
    # ============================================================
    x1_raw, x2_raw = map(imageio.imread, args.input)

    class StaticCenterCrop(object):
        def __init__(self, image_size, crop_size):
            self.th, self.tw = crop_size
            self.h, self.w = image_size
            print(self.th, self.tw, self.h, self.w)
        def __call__(self, img):
            return img[(self.h-self.th)//2:(self.h+self.th)//2, (self.w-self.tw)//2:(self.w+self.tw)//2,:]

    x1_raw = np.array(x1_raw)
    x2_raw = np.array(x2_raw)

    if args.crop_shape is not None:
        cropper = StaticCenterCrop(x1_raw.shape[:2], args.crop_shape)
        x1_raw = cropper(x1_raw)
        x2_raw = cropper(x2_raw)
    if args.resize_shape is not None:
        resizer = partial(cv2.resize, dsize = (0,0), dst = args.resize_shape)
        x1_raw, x2_raw = map(resizer, [x1_raw, x2_raw])
    elif args.resize_scale is not None:
        resizer = partial(cv2.resize, dsize = (0,0), fx = args.resize_scale, fy = args.resize_scale)
        x1_raw, x2_raw = map(resizer, [x1_raw, x2_raw])

    x1_raw = x1_raw[np.newaxis,:,:,:].transpose(0,3,1,2)
    x2_raw = x2_raw[np.newaxis,:,:,:].transpose(0,3,1,2)


    x1_raw = torch.Tensor(x1_raw).to(args.device)
    x2_raw = torch.Tensor(x2_raw).to(args.device)
    

    # Forward Pass
    # ============================================================
    with torch.no_grad():
        output_flow, flows = model(x1_raw, x2_raw)
    flow = flows[-1]
    flow = np.array(flow.data).transpose(0,2,3,1).squeeze(0)
    save_flow(args.output, flow)
    flow_vis = vis_flow(flow)
    imageio.imwrite(args.output.replace('.flo', '.png'), flow_vis)
    import matplotlib.pyplot as plt
    plt.imshow(flow_vis)
    plt.show()



def test(args, eval_iter):
    # TODO
    pass



if __name__ == '__main__':
    main()