import torch
import numpy as np

from .models import Detector, save_model, total_loss
from .utils import load_detection_data, DetectionSuperTuxDataset, DataLoader
from . import dense_transforms  # --uncomment when submitting the project
from .ap import PR, point_in_box, point_close, box_iou

'''from models import Detector, save_model
from utils import load_detection_data
import dense_transforms'''

import torch.utils.tensorboard as tb

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def evaluate_model(model, data_loader):
    det = model.eval().to(device)
    pr_box = [PR() for _ in range(3)]
    pr_dist = [PR(is_close=point_close) for _ in range(3)]
    pr_iou = [PR(is_close=box_iou) for _ in range(3)]

    with torch.no_grad():
        for img, *gts in data_loader:
            img = img.to(device)
            detections = det.detect(img)

            for i, gt in enumerate(gts):
                pr_box[i].add(detections[i], gt)
                pr_dist[i].add(detections[i], gt)
                pr_iou[i].add(detections[i], gt)

    ap = [pr_box[i].average_prec for i in range(3)]
    ap_d = [pr_dist[i].average_prec for i in range(3)]
    ap_iou = [pr_iou[i].average_prec for i in range(3)]
    return ap, ap_d, ap_iou


def train(args):
    from os import path
    train_logger, valid_logger = None, None
    loss = total_loss
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'), flush_secs=1)
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'), flush_secs=1)

    """
    Your code here, modify your HW3 code
    Hint: Use the log function below to debug and visualize your model
    """
    model = Detector().to(device)
    if args.continue_training:
        model.load_state_dict(torch.load(path.join(path.dirname(path.abspath(__file__)), 'det.th')))

    # optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=1e-3)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    # w = get_pos_weight_from_data()
    # Define the custom Focal Loss function

    import inspect
    transform = eval(args.transform, {k: v for k, v in inspect.getmembers(dense_transforms) if inspect.isclass(v)})
    train_data = load_detection_data('dense_data/train', num_workers=4, batch_size=args.batch_size, transform=transform)
    valid_data = DetectionSuperTuxDataset('dense_data/valid', min_size=0)
    global_step = 0
    min_loss = float('inf')
    ave_loss = float('inf')
    v_ap_high = [0.0, 0.0, 0.0]
    v_itr = 0
    for epoch in range(args.num_epoch):
        loss_per_epoc = 0.0
        ap, ap_d, ap_iou = [], [], []
        print(f'Going to process epoch: {epoch} with epoch loss {ave_loss} and min_loss {min_loss}')
        model.train()
        # print("before start fetching the data")
        for data in train_data:
            img = data[0].to(device)
            gt = data[1].to(device)
            # Get predicted heatmap and size from the model
            predicted_heatmap = model(img)
            # Calculate total loss using the custom loss function
            loss_val = loss(predicted_heatmap, gt)
            loss_per_epoc = loss_per_epoc + loss_val.item()
            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()
            global_step += 1
            if train_logger:
                train_logger.add_scalar('global_loss_train', loss_val.item(), global_step)
        ave_loss = loss_per_epoc / len(train_data)
        if v_itr >= 20:
            if valid_logger:
                ap, ap_d, ap_iou = evaluate_model(model, valid_data)
                for i in range(3):
                    valid_logger.add_scalar(f'AP_{i}', ap[i], global_step)
                    valid_logger.add_scalar(f'AP_D_{i}', ap_d[i], global_step)
                    valid_logger.add_scalar(f'AP_IOU_{i}', ap_iou[i], global_step)
                print(f'AP is {ap}')
                print(f'AP for Size is  {ap_d}')
                print(f'AP for IOU is  {ap_iou}')
                v_itr = 0
                print(f'model with ap at {ap}')
                if ap[0] > v_ap_high[0]:
                    v_ap_high[0] = ap[0]
                    print(f'saving model with ap at 0 {v_ap_high[0]}')
                    save_model(model)
                if ap[1] > v_ap_high[1]:
                    v_ap_high[1] = ap[1]
                    print(f'saving model with ap at 1 {v_ap_high[1]}')
                    save_model(model)
                if ap[2] > v_ap_high[2]:
                    v_ap_high[2] = ap[2]
                    print(f'saving model with ap at 0 {v_ap_high[2]}')
                    save_model(model)
        v_itr = v_itr + 1
        if min_loss > ave_loss:
            min_loss = ave_loss
            print(f'saving model with min_loss {min_loss}')
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
    parser.add_argument('-n', '--num_epoch', type=int, default=100)
    parser.add_argument('-b', '--batch_size', type=int, default=45)
    parser.add_argument('-lr', '--learning_rate', type=float, default=.01)
    parser.add_argument('-g', '--gamma', type=float, default=0, help="class dependent weight for cross entropy")
    parser.add_argument('-c', '--continue_training', action='store_true')
    parser.add_argument('-t', '--transform',
                        default='Compose([RandomHorizontalFlip(0),ToTensor(),ToHeatmap()])')

    args = parser.parse_args()
    train(args)