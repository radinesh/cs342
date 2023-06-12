from .models import ClassificationLoss, model_factory, save_model
from .utils import accuracy, load_data
import torch.optim as optim
import torch

# Set Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRAIN_PATH = "data/train"
VALID_PATH = "data/valid"


def train(args):
    model = model_factory[args.model]()
    model.to(device)
    """
    Your code here

    """
    best_valid_loss = float('inf')
    best_acc = 0.0
    loss_fn = ClassificationLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate)
    train_loss_hist = []
    train_acc_hist = []
    test_loss_hist = []
    test_acc_hist = []
    for epoch in range(args.num_epochs):
        model.train()
        t_loss = 0.0
        t_acc = 0.0
        train_loader = load_data(TRAIN_PATH, 2, args.batch_size)
        valid_loader = load_data(VALID_PATH, 2, args.batch_size)
        # trainning the linear model
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            t_loss_it = loss.item()
            t_loss += t_loss_it
            acc = accuracy(logits, labels)
            t_acc += acc
        # validating model
        model.eval()
        v_loss = 0.0
        v_acc = 0.0
        with torch.no_grad():
            for images, labels in valid_loader:
                images = images.to(device)
                labels = labels.to(device)
                logits = model(images)
                loss = loss_fn(logits, labels)
                v_loss += loss.item()
                acc = accuracy(logits, labels)
                v_acc += acc
        avg_t_loss = t_loss / len(train_loader)
        avg_v_loss = v_loss / len(valid_loader)
        avg_t_acc = t_acc / len(train_loader)
        avg_v_acc = v_acc / len(valid_loader)
        #print(f'at epoch: {epoch} average train loss  and accuracy is {avg_t_loss, avg_t_acc} \n')
        #print(f'at epoch: {epoch} average validation loss  and accuracy is {avg_v_loss, avg_v_acc} \n')

        if avg_v_loss < best_valid_loss:
            best_valid_loss = avg_v_loss
            save_model(model)
        if avg_v_acc > best_acc:
            best_acc = avg_v_acc
            save_model(model)
        print(f'at epoch: {epoch} best loss and accuracy is {best_valid_loss, best_acc} \n')
        train_loss_hist.append(avg_t_loss)
        train_acc_hist.append(avg_t_acc)
        test_loss_hist.append(avg_v_loss)
        test_acc_hist.append(avg_v_acc)
    # raise NotImplementedError('train')
    # save_model(model)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    # Put custom arguments here
    parser.add_argument('-m', '--model', type=str, choices=['linear', 'mlp'], default='linear')
    # Put custom arguments here
    parser.add_argument('-b', '--batch_size', type=int, default=28)
    parser.add_argument('-s', '--manual_seed', type=int, default=42)
    parser.add_argument('-ne', '--num_epochs', type=int, default=90)
    parser.add_argument('-l', '--learning_rate', type=float, default=0.001)
    parser.add_argument('-w', '--weight_decay', type=float, default=1e-5)
    parser.add_argument('-p', '--save_path', type=str, default='linear_model.pth')
    args = parser.parse_args()
    train(args)
