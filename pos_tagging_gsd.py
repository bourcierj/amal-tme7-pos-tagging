"""Training a POS-tagger on the French GSD dataset."""
from tqdm import tqdm

import torch
import torch.nn as nn

from utils import CheckpointState, EarlyStopper

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(checkpoint, criterion, train_loader, val_loader, epochs, patience=None, clip=None,
          summary_writer=None):

    print("Beginning of training.")
    print("Training on", 'GPU' if device.type == 'cuda' else 'CPU', '\n')
    net, optimizer = checkpoint.model, checkpoint.optimizer
    if patience is not None:
        early_stopper = EarlyStopper(patience)
    writer = summary_writer
    min_loss = float('inf')
    iteration = 1

    def train_epoch():
        nonlocal iteration
        epoch_loss = 0.
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{epochs}', dynamic_ncols=True)  # progress bar
        net.train()
        for data, lengths, target in pbar:

            data, lengths, target = data.to(device), lengths.to(device), \
                target.to(device)
            batch_size, seq_length = tuple(data.size())
            # reset gradients
            optimizer.zero_grad()
            output = net(data, lengths)
            # flatten output and target to compute loss on individual words
            target = target.view(-1)
            output = output.view(target.size(0), -1)
            loss = criterion(output, target)
            epoch_loss += loss.item()
            pbar.set_postfix(loss=f'{loss.item():.4e}')
            if writer:
                writer.add_scalar('Train/Iteration Loss', loss.item(), iteration)
            # compute gradients, update parameters
            loss.backward()
            if clip is not None:
                # gradient clipping helps prevent the exploding gradient problem in RNNs / LSTMs
                torch.nn.utils.clip_grad_norm_(net.parameters(),
                                               max_norm=clip)
            optimizer.step()
            iteration += 1

        epoch_loss /= len(train_loader)
        print(f'Epoch {epoch}/{epochs}, mean loss: {epoch_loss:.4e}')
        if writer:
            writer.add_scalar('Train/Epoch Loss', epoch_loss, epoch)
        return epoch_loss

    def evaluate_epoch():
        net.eval()
        correct = 0
        total_preds = 0
        val_loss = 0.
        with torch.no_grad():
            for data, lengths, target in val_loader:
                data, lengths, target = data.to(device), lengths.to(device), \
                                        target.to(device)
                batch_size, seq_length = tuple(data.size())
                output = net(data, lengths)
                # flatten output and target to compute loss on individual words
                target = target.view(-1)
                output = output.view(target.size(0), -1)
                loss = criterion(output, target)
                val_loss += loss.item()
                # predict the tags
                _, pred = output.max(1)
                mask = target != 0
                # counts the number of correct predictions
                correct += pred[mask].eq(target[mask]).sum().item()
                total_preds += len(mask)

        val_loss /= len(val_loader)
        acc = correct / total_preds
        print(f'Epoch {epoch}/{epochs}, validation mean loss: {val_loss:.4e}, '
              f'validation accuracy: {acc:.4f}')
        if writer:
            writer.add_scalar('Train/Val Loss', val_loss, epoch)
            writer.add_scalar('Train/Val Accuracy', acc, epoch)
        return val_loss, acc

    losses = list()
    begin_epoch = checkpoint.epoch
    for epoch in range(begin_epoch, epochs+1):

        train_epoch()
        loss, acc = evaluate_epoch()
        checkpoint.epoch += 1
        if loss < min_loss:
            min_loss = loss
            checkpoint.save('_best')
        checkpoint.save()
        losses.append(loss)
        if patience is not None:
            early_stopper.add(loss, epoch)
            if early_stopper.stop():
                print(f"No improvement in {patience} epochs, early stopping.")
                break

    print("\nFinished.")
    print(f"Best loss: {min_loss:.4e}")
    if patience is not None:
        print(f"Best epoch: {early_stopper.min_epoch}")
    return losses


if __name__ == '__main__':

    torch.manual_seed(42)

    from torch.utils.data import DataLoader
    import torch.optim as optim
    from torch.utils.tensorboard import SummaryWriter

    from datamaestro import prepare_dataset

    from tagger import Tagger
    from pos_tagging_data import VocabularyTagging, TaggingDataset

    ds = prepare_dataset('org.universaldependencies.french.gsd')

    words = VocabularyTagging(True)
    tags = VocabularyTagging(False)
    train_dataset = TaggingDataset(ds.files["train"], words, tags, True)
    val_dataset = TaggingDataset(ds.files["dev"], words, tags, True)

    embedding_size = 50
    hidden_size = 50
    lr = 0.001
    batch_size = 128
    epochs = 10
    patience = None
    clip = None  # clips gradients norm

    kwargs = dict(num_workers=torch.multiprocessing.cpu_count(),
                  pin_memory=(device.type == 'cuda'))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              collate_fn=TaggingDataset.collate, **kwargs)

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True,
                            collate_fn=TaggingDataset.collate, **kwargs)

    net = Tagger(len(words), len(tags), embedding_size, hidden_size)
    net = net.to(device)

    # In order to exclude losses computed on null entries (zero),
    # set ignore_index=0 for the loss criterion
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(net.parameters(), lr=lr)

    tensorboard_logdir = './runs/POS-Tagging-1'
    writer = SummaryWriter(tensorboard_logdir)
    savepath = './pos-tagger-checkpt.pt'

    checkpoint = CheckpointState(net, optimizer, savepath=savepath)

    losses = train(checkpoint, criterion, train_loader, val_loader, epochs,
                   patience=patience, clip=clip, summary_writer=writer)
