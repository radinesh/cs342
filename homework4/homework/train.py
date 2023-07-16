import torch
import numpy as np

from .models import Detector, save_model, total_loss
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
    loss = total_loss
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
    # Define the custom Focal Loss function

    import inspect
    transform = eval(args.transform, {k: v for k, v in inspect.getmembers(dense_transforms) if inspect.isclass(v)})
    train_data = load_detection_data('dense_data/train', num_workers=4, transform=transform)
    valid_data = load_detection_data('dense_data/valid', num_workers=4)
    global_step = 0
    loss_per_epoc = 100.00
    for epoch in range(args.num_epoch):
        print(f'Going to process epoch: {epoch} with loss {loss_per_epoc}')
        model.train()
        print("before start fetching the data")
        for data in train_data:
            img = data[0].to(device)
            gt = data[1].to(device)
            st = data[2].to(device)
            # Get predicted heatmap and size from the model
            predicted_heatmap, predicted_size = model(img)
            # Calculate total loss using the custom loss function
            loss_val = loss(predicted_heatmap, predicted_size, gt, st)
            loss_per_epoc=loss_val
            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()
            global_step += 1

            if train_logger:
                train_logger.add_scalar('global_loss', loss_val, global_step)
                log(train_logger, img, gt, loss_val, global_step)
                # train_logger.add_scalar('average_accuracy', conf.average_accuracy, global_step)
                # train_logger.add_scalar('iou', conf.iou, global_step)
        save_model(model)

        ''' model.eval()
            for vdata in valid_data:
                img = vdata[0].to(device)
                gt = vdata[1].to(device)
                st = vdata[2].to(device)
                # Get predicted heatmap and size from the model
                predicted_heatmap, predicted_size = model(img)

            if valid_logger:
                valid_logger.add_scalar('global_accuracy', val_conf.global_accuracy, global_step)
                valid_logger.add_scalar('average_accuracy', val_conf.average_accuracy, global_step)
                valid_logger.add_scalar('iou', val_conf.iou, global_step)

            if valid_logger is None or train_logger is None:
                print('epoch %-3d \t acc = %0.3f \t val acc = %0.3f \t iou = %0.3f \t val iou = %0.3f' %
                      (epoch, conf.global_accuracy, val_conf.global_accuracy, conf.iou, val_conf.iou))
            save_model(model) '''

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
    parser.add_argument('-n', '--num_epoch', type=int, default=110)
    parser.add_argument('-lr', '--learning_rate', type=float, default=.09)
    parser.add_argument('-g', '--gamma', type=float, default=0, help="class dependent weight for cross entropy")
    parser.add_argument('-c', '--continue_training', action='store_true')
    parser.add_argument('-t', '--transform',
                        default='Compose([RandomHorizontalFlip(0),ToTensor(),ToHeatmap()])')

    args = parser.parse_args()
    train(args)
