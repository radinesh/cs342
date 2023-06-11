from .models import ClassificationLoss, model_factory, save_model
from .utils import accuracy, load_data

# Set Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRAIN_PATH = "data/train"
VALID_PATH = "data/valid"

def train(args):
    model = model_factory[args.model]()

    """
    Your code here

    """
    best_valid_loss = float('inf')
    loss_fn = ClassificationLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate)

    for epoch in range(args.num_epochs):
        model.train()
        t_loss = 0.0
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
            t_loss += loss.item()
        # validating model
        model.eval()
        v_loss = 0.0
        with torch.no_grad():
            for images, labels in valid_loader:
                images = images.to(device)
                labels = labels.to(device)
                logits = model(images)
                loss = loss_fn(logits, labels)
                v_loss += loss.item()
        avg_t_loss = t_loss / len(train_loader)
        avg_v_loss = v_loss / len(valid_loader)
        if avg_v_loss < best_valid_loss:
            best_valid_loss = avg_v_loss
            save_model(model)
    # raise NotImplementedError('train')
    save_model(model)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('-m', '--model', choices=['linear', 'mlp'], default='linear')
    # Put custom arguments here
    parser.add_argument('-m', '--model', type=str, choices=['linear', 'mlp'], default='linear')
    # Put custom arguments here
    parser.add_argument('-b', '--batch_size', type=int, default=128)
    parser.add_argument('-s', '--manual_seed', type=int, default=42)
    parser.add_argument('-ne', '--num_epochs', type=int, default=10)
    parser.add_argument('-l', '--learning_rate', type=float, default=0.001)
    parser.add_argument('-w', '--weight_decay', type=float, default=1e-5)
    parser.add_argument('-p', '--save_path', type=str, default='linear_model.pth')
    args = parser.parse_args()
    train(args)
