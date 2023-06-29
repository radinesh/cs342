from .models import CNNClassifier,ClassificationLoss, save_model
from .utils import accuracy, load_data
import torch
import torch.utils.tensorboard as tb
import torch.optim as optim
# Set Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRAIN_PATH = "data/train"
VALID_PATH = "data/valid"


def train(args):
    from os import path
    model = CNNClassifier()
    model.to(device)
    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'), flush_secs=1)
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'), flush_secs=1)

    """
    Your code here, modify your HW1 / HW2 code
    """
    best_valid_loss = float('inf')
    best_acc = 0.0
    global_step = 0
    loss_fn = ClassificationLoss()
    optimizer = optim.Adam(model.parameters(),
                           betas=(0.9, 0.999), lr=args.learning_rate, eps=1e-08, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [50, 75], gamma=0.1)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max')
    for epoch in range(args.num_epochs):
        print(f'going to process epoch {epoch} from total epoch {args.num_epochs}')
        model.train()
        # accuracies = []
        t_loss = 0.0
        t_acc = 0.0
        train_loader = load_data(TRAIN_PATH, 2, args.batch_size)
        valid_loader = load_data(VALID_PATH, 2, args.batch_size)
        # trainning the linear model
        for image, label in train_loader:
            image = image.to(device)
            label = label.to(device)
            logits = model(image)
            loss = loss_fn(logits, label)
            # accuracies.extend(accuracy(logits, label).detach().cpu().numpy())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            t_loss_it = loss.item()
            t_loss += t_loss_it
            acc = accuracy(logits, label)
            t_acc += acc
        scheduler.step(t_acc/len(train_loader))
        # validating model
        model.eval()
        v_loss = 0.0
        v_acc = 0.0
        with torch.no_grad():
            for image, label in valid_loader:
                image = image.to(device)
                label = label.to(device)
                logits = model(image)
                loss = loss_fn(logits, label)
                v_loss += loss.item()
                acc = accuracy(logits, label)
                v_acc += acc
        avg_t_loss = t_loss / len(train_loader)
        avg_v_loss = v_loss / len(valid_loader)
        avg_t_acc = t_acc / len(train_loader)
        avg_v_acc = v_acc / len(valid_loader)
        train_logger.add_scalar('loss', avg_t_loss, global_step=global_step)
        train_logger.add_scalar('accuracy', avg_t_acc, global_step=global_step)
        valid_logger.add_scalar('loss', avg_v_loss, global_step=global_step)
        valid_logger.add_scalar('accuracy', avg_v_acc, global_step=global_step)
        print(f'at epoch: {epoch} average train loss  and accuracy is {avg_t_loss, avg_t_acc} \n')
        print(f'at epoch: {epoch} average validation loss  and accuracy is {avg_v_loss, avg_v_acc} \n')

        if avg_v_loss < best_valid_loss:
            best_valid_loss = avg_v_loss
            save_model(model)
        if avg_v_acc > best_acc:
            best_acc = avg_v_acc
            save_model(model)
        print(f'at epoch: {epoch} best loss and accuracy is {best_valid_loss, best_acc} \n')


if __name__ == '__main__':
    import argparse


    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir',type=str, default='./ccnLogs')
    # Put custom arguments here
    parser.add_argument('-b', '--batch_size', type=int, default=28)
    parser.add_argument('-s', '--manual_seed', type=int, default=42)
    parser.add_argument('-ne', '--num_epochs', type=int, default=50)
    parser.add_argument('-l', '--learning_rate', type=float, default=0.001)
    parser.add_argument('-w', '--weight_decay', type=float, default=1e-08)
    args = parser.parse_args()
    train(args)
