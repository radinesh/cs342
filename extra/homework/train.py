import torch
import torch.nn as nn
from .models import TCN, save_model,stack_param
from .utils import load_data,one_hot, SpeechDataset
import numpy as np

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print('device = ', device)


'''def make_random_batch(batch_size, seq_len, one_hot_data):
    B = []
    for i in range(batch_size):
        s = np.random.choice(one_hot_data.size(1) - seq_len)
        B.append(one_hot[:, s:s + seq_len])
    return torch.stack(B, dim=0)
'''


def train(args):
    from os import path
    import torch.utils.tensorboard as tb
    torch.autograd.set_detect_anomaly(True)
    train_logger = None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'), flush_secs=1)

    """
    Your code here, modify your code from prior assignments
    Hint: SGD might need a fairly high learning rate to work well here

    """
    # Create the network
    model = TCN().to(device)
    # Create the optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=0.5, momentum=0.9)

    # Create the loss
    loss = torch.nn.CrossEntropyLoss()
    data_batch = load_data('data/train.txt', transform=one_hot, max_len=args.seq_len)
    # data = data.to(device)
    # Start training
    global_step = 0
    min_loss = float('inf')
    for iterations in range(args.num_epoch):
        model.train()
        print("Going to process epoch ....", iterations)
        total_loss = 0.0
        for data in data_batch:
            batch_data = data[:, :, :-1].to(device)
            batch_label = batch_data.argmax(dim=1).to(device)
            batch = stack_param(torch.nn.Parameter(torch.rand(28, 1), requires_grad=True), batch_label)
            batch = batch[:, 0, :]
            # print("Model batch size ", batch.shape)
            # print("batch_label shape before ", batch_label.shape)
            batch_label = torch.cat([batch, batch_label], dim=1).long()
            # batch = batch.squeeze(2)
            # print("Model batch size ", batch.shape)
            # print("batch_data shape", batch_data.shape)
            # print("batch_label shape", batch_label.shape)
            o = model(batch_data)
            # print("output shape", o.shape)
            loss_val = loss(o, batch_label)
            total_loss = total_loss + loss_val.item()
            if train_logger is not None:
                train_logger.add_scalar('train/loss', loss_val, global_step=global_step)
            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()
            global_step += 1
        av_loss = total_loss/len(data_batch)
        if train_logger is not None:
            train_logger.add_scalar('train/ average loss', av_loss, global_step=global_step)
        print(f'Epoch [{iterations}], Train Loss: {av_loss:.4f}')
        if min_loss > av_loss:
            min_loss = av_loss
            save_model(model)
            print(f'Saving Model for Epoch [{iterations}], Train Loss: {av_loss:.6f},Min Loss {min_loss:.6f}')
        model.eval()
        data = SpeechDataset('data/valid.txt')
        lls = []
        for s in data:
            ll = model.predict_all(s)
            lls.append(float((ll[:, :-1] * one_hot(s)).sum() / len(s)))
        nll = -np.mean(lls)
        print(f'Accuracy  for  [{iterations}], is : {nll:0.3f}')
        if nll < 1.3:
            break


    # raise NotImplementedError('train')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir', type=str, default='tcn_log')
    # Put custom arguments here
    parser.add_argument('-b', '--batch_size', type=int, default=128)
    parser.add_argument('-n', '--num_epoch', type=int, default=10000)
    parser.add_argument('-s', '--seq_len', type=int, default=256)
    args = parser.parse_args()
    train(args)
