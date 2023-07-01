import torch
import numpy as np

from .models import FCN,ClassificationLoss, save_model
from .utils import load_dense_data, DENSE_CLASS_DISTRIBUTION, ConfusionMatrix
from . import dense_transforms
import torch.utils.tensorboard as tb
import torch.optim as optim
from torchvision import transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRAIN_PATH = "dense_data/train"
VALID_PATH = "dense_data/valid"
def train(args):
    from os import path
    model = FCN()
    model.to(device)
    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'), flush_secs=1)
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'), flush_secs=1)

    """
    Your code here, modify your HW1 / HW2 code
    Hint: Use ConfusionMatrix, ConfusionMatrix.add(logit.argmax(1), label), ConfusionMatrix.iou to compute
          the overall IoU, where label are the batch labels, and logit are the logits of your classifier.
    Hint: If you found a good data augmentation parameters for the CNN, use them here too. Use dense_transforms
    Hint: Use the log function below to debug and visualize your model
    """
    # Define data augmentation transforms
    data_transform = dense_transforms.Compose([
        dense_transforms.Resize(96),
        dense_transforms.RandomHorizontalFlip(),
        dense_transforms.RandomResizedCrop((64, 64)),
        dense_transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        dense_transforms.ToTensor()
    ])

    best_valid_loss = float('inf')
    best_acc = 0.0
    global_step = 0
    loss_fn = ClassificationLoss()  # Use CrossEntropyLoss for dense labeling
    optimizer = optim.Adam(model.parameters(),
                           betas=(0.9, 0.999),
                           lr=args.learning_rate,
                           eps=1e-08,
                           weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max')
    for epoch in range(args.num_epochs):
        print(f'Processing epoch {epoch} out of {args.num_epochs}')
        model.train()
        t_loss = 0.0
        t_iou = 0.0
        t_acc = 0.0
        train_loader = load_dense_data(TRAIN_PATH,args.batch_size, transform=data_transform)
        valid_loader = load_dense_data(VALID_PATH, transform=transforms.ToTensor())
        for image, label in train_loader:
            image = image.to(device)
            label = label.to(device)
            logits = model(image)
            target = label.to(torch.long)
            loss = loss_fn(logits, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            t_loss += loss.item()

            # Compute accuracy and IoU using ConfusionMatrix
            cm = ConfusionMatrix()
            cm.add(logits.argmax(dim=1), label)
            t_acc += cm.class_accuracy.mean()
            t_iou += cm.iou

        scheduler.step(t_acc / len(train_loader))
        model.eval()
        v_loss = 0.0
        v_iou = 0.0
        v_acc = 0.0
        print("going to finish train  process...")
        with torch.no_grad():
            for image, label in valid_loader:
                image = image.to(device)
                label = label.to(device)
                logits = model(image)
                target = label.to(torch.long)
                loss = loss_fn(logits, target)
                print("going to load process Valid loader...")
                v_loss += loss.item()

                # Compute accuracy and IoU using ConfusionMatrix
                cm = ConfusionMatrix()
                cm.add(logits.argmax(dim=1), label)
                v_acc += cm.class_accuracy.mean()
                v_iou += cm.iou

        avg_t_loss = t_loss / len(train_loader)
        avg_v_loss = v_loss / len(valid_loader)
        avg_t_acc = t_acc / len(train_loader)
        avg_v_acc = v_acc / len(train_loader)
        avg_t_iou = t_iou / len(train_loader)
        avg_v_iou = v_iou / len(train_loader)
        train_logger.add_scalar('loss', avg_t_loss, global_step=global_step)
        train_logger.add_scalar('accuracy', avg_t_acc, global_step=global_step)
        train_logger.add_scalar('iou', avg_t_iou, global_step=global_step)
        valid_logger.add_scalar('loss', avg_v_loss, global_step=global_step)
        valid_logger.add_scalar('accuracy', avg_v_acc, global_step=global_step)
        train_logger.add_scalar('iou', avg_t_iou, global_step=global_step)
        print(f'at epoch: {epoch} average train loss,accuracy  and iou is {avg_t_loss, avg_t_acc,avg_t_iou} \n')
        print(f'at epoch: {epoch} average validation loss accuracy  and iou is {avg_v_loss, avg_v_acc,avg_v_iou} \n')

        if avg_v_loss < best_valid_loss:
            best_valid_loss = avg_v_loss
            save_model(model)
        if avg_v_acc > best_acc:
            best_acc = avg_v_acc
            save_model(model)
        print(f'at epoch: {epoch} best loss and accuracy is {best_valid_loss, best_acc} \n')

    save_model(model)


def log(logger, imgs, lbls, logits, global_step):
    """
    logger: train_logger/valid_logger
    imgs: image tensor from data loader
    lbls: semantic label tensor
    logits: predicted logits tensor
    global_step: iteration
    """
    logger.add_image('image', imgs[0], global_step)
    logger.add_image('label', np.array(dense_transforms.label_to_pil_image(lbls[0].cpu()).
                                             convert('RGB')), global_step, dataformats='HWC')
    logger.add_image('prediction', np.array(dense_transforms.
                                                  label_to_pil_image(logits[0].argmax(dim=0).cpu()).
                                                  convert('RGB')), global_step, dataformats='HWC')

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir', type=str, default='./fcnLogs')
    # Put custom arguments here
    parser.add_argument('-b', '--batch_size', type=int, default=8)
    parser.add_argument('-s', '--manual_seed', type=int, default=42)
    parser.add_argument('-ne', '--num_epochs', type=int, default=50)
    parser.add_argument('-l', '--learning_rate', type=float, default=0.001)
    parser.add_argument('-w', '--weight_decay', type=float, default=1e-08)
    args = parser.parse_args()
    train(args)
