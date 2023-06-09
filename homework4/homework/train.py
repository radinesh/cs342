import torch
import numpy as np

from .models import Detector, save_model
from .utils import load_detection_data
from . import dense_transforms  # --uncomment when submitting the project

'''from models import Detector, save_model
from utils import load_detection_data
import dense_transforms'''

import torch.utils.tensorboard as tb


def train(args):
    from os import path
    model = Detector()
    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'), flush_secs=1)
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'), flush_secs=1)

    """
    Your code here, modify your HW3 code
    Hint: Use the log function below to debug and visualize your model
    """

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = Detector().to(device)
    if args.continue_training:
        model.load_state_dict(torch.load(path.join(path.dirname(path.abspath(__file__)), 'det.th')))

    # optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=1e-3)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    # w = get_pos_weight_from_data()
    loss = torch.nn.BCEWithLogitsLoss()
    # raise NotImplementedError('train')
    import inspect
    transform = eval(args.transform, {k: v for k, v in inspect.getmembers(dense_transforms) if inspect.isclass(v)})
    train_data = load_detection_data('dense_data/train', num_workers=0, transform=transform)
    valid_data = load_detection_data('dense_data/valid', num_workers=0)
    global_step = 0
    for epoch in range(args.num_epoch):
        print("Going to process epoch: ", epoch)
        if 1 <= epoch <= 100:
            optimizer.param_groups[0]['lr'] = 0.01  # High learning rate for GT detect loss
        elif 100 < epoch <= 120:
            optimizer.param_groups[0]['lr'] = 0.001  # Low learning rate for GT size loss

        model.train()
        print("before start fetching the data")
        for data in train_data:
            print("going to batch loop..........")
            print("going to batch loop..........img size ", data[0].shape)
            logit = model(data[0])
            print(f'logit.shape {logit.shape} and gt_detect.shape {data[1].shape} gt_size shape {data[2].shape}')
            loss_val = loss(logit, data[1])
            # Calculate loss based on the condition
            '''if epoch <= 100:
                loss_val = loss(logit[0], data[1])
            else:
                loss_val = loss(logit, data[2])
            if train_logger is not None and global_step % 100 == 0:
                log(train_logger, data[0], data[1], logit, global_step)

            if train_logger is not None:
                train_logger.add_scalar('loss', loss_val, global_step)'''
            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()
            global_step += 1

        save_model(model)


def log(logger, imgs, gt_det, det, global_step):
    """
    logger: train_logger/valid_logger
    imgs: image tensor from data loader
    gt_det: ground-truth object-center maps
    det: predicted object-center heatmaps
    global_step: iteration
    """
    logger.add_images('image', imgs[:16], global_step)
    logger.add_images('label', gt_det[:16], global_step)
    logger.add_images('pred', torch.sigmoid(det[:16]), global_step)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir', type=str, default='./dthLogs')
    # Put custom arguments here
    parser.add_argument('-n', '--num_epoch', type=int, default=2)
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3)
    parser.add_argument('-g', '--gamma', type=float, default=0, help="class dependent weight for cross entropy")
    parser.add_argument('-c', '--continue_training', action='store_true')
    parser.add_argument('-t', '--transform',
                        default='Compose([RandomHorizontalFlip(0),ToTensor(),ToHeatmap()])')

    args = parser.parse_args()
    train(args)
