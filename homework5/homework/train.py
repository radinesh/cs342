from .planner import Planner, save_model, spatial_argmax
import torch
import torchvision
import torch.utils.tensorboard as tb
import numpy as np
from .utils import load_data
from . import dense_transforms as dt

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def train(args):
    from os import path
    model = Planner().to(device)
    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'))

    # Define your dataset and data loaders
    '''import inspect
    transform = eval(args.transform,
                     {k: v for k, v in inspect.getmembers(torchvision.transforms) if inspect.isclass(v)})'''
    transform = [
        dt.RandomHorizontalFlip(flip_prob=0.5),
        dt.ColorJitter(brightness=(0.09999999999999998, 1.9), contrast=(0.09999999999999998, 1.9),
                       saturation=(0.09999999999999998, 1.9), hue=(-0.1, 0.1)),
        dt.ToTensor()
    ]
    print("transform is ", transform)
    train_loader = load_data('drive_data')

    # Define the optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    criterion = torch.nn.MSELoss()

    # Training loop
    global_step = 0
    min_loss = float('inf')
    for epoch in range(args.num_epoch):
        print("Going to process epoch ....", epoch)
        total_loss = 0.0
        model.train()
        for images, labels in train_loader:
            optimizer.zero_grad()
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            aim_points = model(images)

            # Calculate loss
            loss = criterion(aim_points, labels)
            total_loss = total_loss + loss.item()
            # Backward and optimize
            loss.backward()
            optimizer.step()
            global_step += 1
            # Log the training progress
            if train_logger is not None:
                log(train_logger, images, labels, aim_points, global_step)
                train_logger.add_scalar('loss', total_loss / len(train_loader), global_step)
        average_loss = total_loss / len(train_loader)
        print(f'Epoch [{epoch}], Train Loss: {average_loss:.4f}')
        if min_loss > average_loss:
            min_loss = average_loss
            save_model(model)
            print(f'Saving Model for Epoch [{epoch}], Train Loss: {average_loss:.6f},Min Loss {min_loss}')


def log(logger, img, label, pred, global_step):
    """
    logger: train_logger/valid_logger
    img: image tensor from data loader
    label: ground-truth aim point
    pred: predited aim point
    global_step: iteration
    """
    import matplotlib.pyplot as plt
    import torchvision.transforms.functional as TF
    fig, ax = plt.subplots(1, 1)
    ax.imshow(TF.to_pil_image(img[0].cpu()))
    WH2 = np.array([img.size(-1), img.size(-2)]) / 2
    ax.add_artist(plt.Circle(WH2 * (label[0].cpu().detach().numpy() + 1), 2, ec='g', fill=False, lw=1.5))
    ax.add_artist(plt.Circle(WH2 * (pred[0].cpu().detach().numpy() + 1), 2, ec='r', fill=False, lw=1.5))
    logger.add_figure('viz', fig, global_step)
    del ax, fig


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir', type=str, default='./plogs')
    # Put custom arguments here
    parser.add_argument('-n', '--num_epoch', type=int, default=75)
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-2)
    parser.add_argument('-c', '--continue_training', action='store_true')
    parser.add_argument('-t', '--transform',
                        default='Compose([ColorJitter(0.9, 0.9, 0.9, 0.1),RandomHorizontalFlip(), ToTensor()])')
    args = parser.parse_args()
    train(args)
