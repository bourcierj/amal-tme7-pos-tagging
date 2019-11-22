"""Training a POS-tagger on the French GSD dataset."""

from tqdm import tqdm
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(checkpoint, criterion, train_loader, val_loader, epochs, patience=None, clip=None,
          writer=None):
    """Full training loop"""

    print("Training on", 'GPU' if device.type == 'cuda' else 'CPU', '\n')
    net, optimizer = checkpoint.model, checkpoint.optimizer
    if patience is not None:
        early_stopper = EarlyStopper(patience)
    if clip is None:  # no gradient clipping
        clip = float('inf')
    min_loss = float('inf')
    iteration = 1

    def train_epoch():
        """
        Returns:
            The epoch loss
        """
        nonlocal iteration
        epoch_loss = 0.
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{epochs}', dynamic_ncols=True)  # progress bar
        net.train()
        for data, lengths, target in pbar:

            data, lengths, target = data.to(device), lengths.to(device), \
                target.to(device)
            seq_length, batch_size = tuple(data.size())
            # reset gradients
            optimizer.zero_grad()
            output = net(data, lengths)
            # flatten output and target to compute loss on individual words
            loss = criterion(output.view(batch_size*seq_length, -1), target.view(-1))
            epoch_loss += loss.item()
            pbar.set_postfix(loss=f'{loss.item():.4e}')
            if writer:
                writer.add_scalar('Iteration_loss', loss.item(), iteration)
            # compute gradients, update parameters
            loss.backward()
            # Gradient clipping helps prevent the exploding gradient problem in RNNs
            # clip the gradients to the given clip value (+inf if not specified)
            # and also return the total norm of parameters
            total_norm = torch.nn.utils.clip_grad_norm_(net.parameters(),
                                                        max_norm=clip)
            if writer:
                writer.add_scalar('Total_norm_of_parameters_gradients', total_norm, iteration)

            optimizer.step()
            iteration += 1

        epoch_loss /= len(train_loader)
        print(f"Epoch {epoch}/{epochs}, Mean loss: {epoch_loss:.4e}")
        return epoch_loss

    def evaluate_epoch(loader, role='Val'):
        """
        Args:
            loader (torch.utils.data.DataLoader): either the train of validation loader
            role (str): either 'Val' or 'Train'
        Returns:
            Tuple containing mean loss and accuracy
        """
        net.eval()
        correct = 0
        total_preds = 0
        mean_loss = 0.
        with torch.no_grad():
            for data, lengths, target in loader:
                data, lengths, target = data.to(device), lengths.to(device), \
                    target.to(device)
                seq_length, batch_size = tuple(data.size())
                output = net(data, lengths)
                # flatten output and target to compute loss on individual words
                target = target.view(-1)
                output = output.view(batch_size*seq_length, -1)
                loss = criterion(output, target)
                mean_loss += loss.item()
                # predict the tags
                _, pred = F.log_softmax(output, 1).max(1)
                mask = target != 0
                # counts the number of correct predictions
                correct += pred[mask].eq(target[mask]).sum().item()
                total_preds += mask.sum().item()

        mean_loss /= len(loader)
        acc = correct / total_preds
        print(f"Epoch {epoch}/{epochs}, {role} mean loss: {mean_loss:.4e}, "
              f"{role} accuracy: {acc:.4f}")
        if writer:
            writer.add_scalar(f"{role}_epoch_loss", mean_loss, epoch)
            writer.add_scalar(f"{role}_epoch_accuracy", acc, epoch)
        return mean_loss, acc

    losses = list()
    begin_epoch = checkpoint.epoch
    for epoch in range(begin_epoch, epochs+1):

        train_epoch()
        evaluate_epoch(train_loader, 'Train')
        loss, acc = evaluate_epoch(val_loader, 'Val')
        checkpoint.epoch += 1
        if loss < min_loss:
            min_loss = loss
            best_acc = acc
            checkpoint.save('_best')
        checkpoint.save()
        losses.append(loss)
        if patience is not None:
            early_stopper.add(loss, epoch)
            if early_stopper.stop():
                print(f"No improvement in {patience} epochs, early stopping.")
                break

    print("\nFinished.")
    print(f"Best validation loss: {min_loss:.4e}")
    print(f"With accuracy: {best_acc}")
    if patience is not None:
        print(f"Best epoch: {early_stopper.min_epoch}")
    return losses


if __name__ == '__main__':

    def parse_args():
        """Parse command line arguments."""
        parser = argparse.ArgumentParser(
            description="Trains a POS-tagger on the French-GSD dataset.")

        parser.add_argument('--no-tensorboard', action='store_true')
        parser.add_argument('--batch-size', default=128, type=int)
        parser.add_argument('--lr', default=0.001, type=float)
        parser.add_argument('--epochs', default=20, type=int)
        parser.add_argument('--patience', default=None, type=int)
        parser.add_argument('--clip', default=None, type=float)
        parser.add_argument('--embedding-size', default=30, type=int)
        parser.add_argument('--hidden-size', default=30, type=int)
        parser.add_argument('--num-layers', default=1, type=int)
        parser.add_argument('--dropout', default=0, type=float)
        parser.add_argument('--bidirectional', action='store_true')
        return parser.parse_args()

    torch.manual_seed(42)
    args = parse_args()

    import torch.optim as optim
    from torch.utils.tensorboard import SummaryWriter

    from tagger import Tagger
    from pos_tagging_data import *

    train_loader, val_loader, _, words, tags = \
        get_dataloaders_and_vocabs(args.batch_size)

    net = Tagger(len(words), len(tags), args.embedding_size, args.hidden_size,
                 args.num_layers, args.dropout, args.bidirectional)
    net = net.to(device)

    # In order to exclude losses computed on null entries (zero),
    # set ignore_index=0 for the loss criterion
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(net.parameters(), lr=args.lr)

    ignore_keys = {'no_tensorboard'}
    # get hyperparameters with values in a dict
    hparams = {key.replace('_','-'): val for key, val in vars(args).items()
               if key not in ignore_keys}
    # generate a name for the experiment
    expe_name = '_'.join([f"{key}={val}" for key, val in hparams.items()])
    # path where to save the model
    savepath = Path('./checkpoints/checkpt.pt')
    # Tensorboard summary writer
    if args.no_tensorboard:
        writer = None
    else:
        writer = SummaryWriter(comment='__POS-Tagging-GRU__'+expe_name, flush_secs=10)
        # log sample data and net graph in tensorboard
        data, lengths, target = next(iter(train_loader))
        writer.add_graph(net, (data, lengths))

    checkpoint = CheckpointState(net, optimizer, savepath=savepath)

    train(checkpoint, criterion, train_loader, val_loader, args.epochs,
          patience=args.patience, clip=args.clip, writer=writer)
