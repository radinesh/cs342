from os import path
import torch
import numpy as np
import torch.utils.tensorboard as tb


def test_logging(train_logger, valid_logger):

    """
    Your code here.
    Finish logging the dummy loss and accuracy
    Log the loss every iteration, the accuracy only after each epoch
    Make sure to set global_step correctly, for epoch=0, iteration=0: global_step=0
    Call the loss 'loss', and accuracy 'accuracy' (no slash or other namespace)
    """

    # This is a strongly simplified training loop
    # Start training
    global_step = 0
    for epoch in range(10):
        torch.manual_seed(epoch)
        dummy_validation_accuracy_list = []
        dummy_train_accuracy_list = []
        for iteration in range(20):
            dummy_train_loss = 0.9**(epoch+iteration/20.)
            dummy_train_accuracy = epoch/10. + torch.randn(10)
            dummy_train_accuracy_list.extend(dummy_train_accuracy.cpu().detach().numpy())
            train_logger.add_scalar('train/loss', dummy_train_loss, global_step=global_step)
            global_step += 1

            # raise NotImplementedError('Log the training loss')
        #raise NotImplementedError('Log the training accuracy')
        print(f' Train Accuracy list is : {np.mean(dummy_train_accuracy_list)}')
        train_logger.add_scalar('train/accuracy', np.mean(dummy_train_accuracy_list), global_step=epoch)
        torch.manual_seed(epoch)
        for iteration in range(10):
            dummy_validation_accuracy = epoch / 10. + torch.randn(10)
            dummy_validation_accuracy_list.extend(dummy_validation_accuracy.cpu().detach().numpy())
        print(f' Valid Accuracy list is : {np.mean(dummy_validation_accuracy_list)}')
        valid_logger.add_scalar('valid/accuracy', np.mean(dummy_validation_accuracy_list), global_step=epoch)
        #raise NotImplementedError('Log the validation accuracy')


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('log_dir')
    args = parser.parse_args()
    train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'))
    valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'test'))
    test_logging(train_logger, valid_logger)
