def parse():
    import argparse
    parser = argparse.ArgumentParser(description='Structure from Motion Learner training on KITTI and CityScapes Dataset',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # mode selection
    # ============================================================
    parser.add_argument('--train', action = 'store_true')
    parser.add_argument('--predict', action = 'store_true')
    parser.add_argument('--test', action = 'store_true')



    # mode=train args
    # ============================================================
    from datetime import datetime
    parser.add_argument('--log_dir', default = datetime.now().strftime('%Y%m%d-%H%M%S'))
    parser.add_argument('--dataset_dir', type = str)
    parser.add_argument('--dataset', type = str)
    parser.add_argument('--weights', nargs = '+', default = [0.32, 0.08, 0.02, 0.01, 0.005])
    parser.add_argument('--epsilon', default = 0.02)
    parser.add_argument('--q', default = 0.4)
    parser.add_argument('--gamma', default = 4e-4)
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
    from model import Net
    model = Net(args)

    # TODO: change optimizer to S_long & S_fine (same as flownet2)
    optimizer = torch.optim.Adam(parameters, args.lr,
                                 betas=(args.momentum, args.beta),
                                 weight_decay=args.weight_decay)
    # build criterion
    criterion = get_criterion(args)
    model_parameters = model.parameters()
    pass

    
    # Prepare Dataloader
    # ============================================================
    from dataset import MPISintel
    import torch
    from torch.autograd import Variable
    from torch.utils.data import DataLoader
    import torch.nn.functional as F
    train_dataset = MPISintel('data_train.txt')
    eval_dataset = MPISintel('data_test.txt')
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
    from logger import Logger
    logger = Logger(args.log_dir)


    for batch_idx, (data, target) in enumerate(train_loader):

        
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
        from losses import compute_loss
        loss = criterion(args, flow_pyramid, flow_gt_pyramid, model_parameters)


        
        # Do step
        # ============================================================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        
        # Collect Summaries & Output Logs
        # ============================================================
        
        # flow output on each level
        if step % args.summary_interval == 0:
            # add scalar summary


            # add image summary
            # logger.image_summary()
            pass
        
        if step % args.log_interval == 0:
            print(f'Step [{step}/{args.total_step}], Loss: {loss:.4f}, EPE: {epe:.4f}')


def predict(args):
    # TODO
    # Build Model
    # ============================================================
    from model import Net
    model = Net(args)
    args.input
    args.output


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