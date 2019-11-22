"""
Prints POS-tagger results of loss and accuracy on train, dev and test set of
French-GSD dataset
"""

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def evaluate(net, criterion, loader, role='Test'):
    """Evaluates a trained POS-tagger on a dataset"""
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
    print(f"{role} mean loss: {mean_loss:.4e}, {role} accuracy: {acc:.4f}")
    return mean_loss, acc


if __name__ == '__main__':

    def parse_args():
        """Parse command line arguments."""
        parser = argparse.ArgumentParser(
            description="Evaluates a POS-tagger on the test set of the French-GSD "
                        "dataset.")
        parser.add_argument("--saved-path",
                            default='./checkpoints/saved/checkpt.pt',
                            type=str)
        return parser.parse_args()

    torch.manual_seed(42)
    args = parse_args()

    from tagger import Tagger
    from pos_tagging_data import *
    from utils import *

    batch_size = 128
    train_loader, val_loader, test_loader, words, tags = \
        get_dataloaders_and_vocabs(batch_size)

    criterion = nn.CrossEntropyLoss(ignore_index=0)

    # load model from saved checkpoint
    net = Tagger(len(words), len(tags), embedding_size=30, hidden_size=30)
    checkpoint = CheckpointState(net, savepath=args.saved_path)
    checkpoint.load()
    net = net.to(device)

    # evaluate the model on the train, validation and test set,
    # printing loss and accuracy
    evaluate(net, criterion, train_loader, 'Train')
    evaluate(net, criterion, val_loader, 'Val')
    evaluate(net, criterion, test_loader, 'Test')
